"""Upstream Scene Change Analyzer.

Detects field changes between local scenes and their stash-box counterparts.
Handles both simple fields (title, date, etc.) and relational fields
(studio, performers, tags) with set-based comparison.
"""

import logging
from typing import Optional

from .base_upstream import BaseUpstreamAnalyzer
from settings import get_setting
from upstream_field_mapper import (
    normalize_upstream_scene,
    diff_scene_fields,
    DEFAULT_SCENE_FIELDS,
    SCENE_FIELD_LABELS,
)

logger = logging.getLogger(__name__)

GENDER_SETTING_KEY_BY_CANONICAL: dict[str, str] = {
    "FEMALE": "upstream_scene_gender_female_enabled",
    "MALE": "upstream_scene_gender_male_enabled",
    "TRANSGENDER_FEMALE": "upstream_scene_gender_transgender_female_enabled",
    "TRANSGENDER_MALE": "upstream_scene_gender_transgender_male_enabled",
    "INTERSEX": "upstream_scene_gender_intersex_enabled",
    "NON_BINARY": "upstream_scene_gender_non_binary_enabled",
    "UNKNOWN": "upstream_scene_gender_unknown_enabled",
}


def _normalize_performer_gender(gender: Optional[str]) -> Optional[str]:
    """Normalize performer gender values to canonical enum strings."""
    if not gender:
        return None

    normalized = str(gender).strip().upper().replace("-", "_").replace(" ", "_")

    aliases = {
        "NONBINARY": "NON_BINARY",
        "TRANS_FEMALE": "TRANSGENDER_FEMALE",
        "TRANS_MALE": "TRANSGENDER_MALE",
        "UNSPECIFIED": "UNKNOWN",
    }
    return aliases.get(normalized, normalized)


def _normalize_endpoint_for_compare(endpoint: Optional[str]) -> str:
    """Normalize endpoint values for resilient stash_id endpoint matching."""
    if not endpoint:
        return ""
    value = str(endpoint).strip().lower().rstrip("/")
    if value.startswith("https://"):
        value = value[len("https://"):]
    elif value.startswith("http://"):
        value = value[len("http://"):]
    if value.endswith("/graphql"):
        value = value[:-len("/graphql")]
    return value


def _endpoint_matches(left: Optional[str], right: Optional[str]) -> bool:
    """Compare two stash-box endpoints while ignoring formatting differences."""
    return _normalize_endpoint_for_compare(left) == _normalize_endpoint_for_compare(right)


def _normalize_stashbox_id(value: Optional[str]) -> str:
    """Normalize stash-box IDs for stable case-insensitive comparisons."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _has_scene_changes(result: dict) -> bool:
    """Check if a scene diff result has any actual changes."""
    if result.get("changes"):
        return True
    if result.get("studio_change") is not None:
        return True
    pc = result.get("performer_changes", {})
    # Scene-level performer aliases are not writable in local Stash, so
    # alias-only differences are intentionally ignored here.
    if pc.get("added") or pc.get("removed"):
        return True
    tc = result.get("tag_changes", {})
    if tc.get("added") or tc.get("removed"):
        return True
    return False


def _normalize_name(value: Optional[str]) -> str:
    """Normalize names for stable case-insensitive comparisons."""
    if not value:
        return ""
    return str(value).strip().lower()


class UpstreamSceneAnalyzer(BaseUpstreamAnalyzer):
    """Detect upstream changes to scenes linked to stash-box endpoints."""

    type = "upstream_scene_changes"
    _current_endpoint: str = ""
    _performer_name_lookup: dict[str, str] | None = None
    _tag_name_lookup: dict[str, str] | None = None
    _studio_name_lookup: dict[str, str] | None = None

    @property
    def entity_type(self) -> str:
        return "scene"

    async def _build_name_lookups(self):
        """Build name→local_id lookup dicts for auto-matching added entities."""
        if self._performer_name_lookup is not None:
            return  # Already built for this run

        # Performers: name + aliases
        self._performer_name_lookup = {}
        all_performers = await self.stash.get_all_performers()
        for p in all_performers:
            pid = str(p["id"])
            name = (p.get("name") or "").strip().lower()
            if name:
                self._performer_name_lookup[name] = pid
            for alias in (p.get("alias_list") or []):
                alias_lower = alias.strip().lower()
                if alias_lower:
                    # Don't overwrite a primary name match
                    self._performer_name_lookup.setdefault(alias_lower, pid)

        # Tags: name only (allTags doesn't include aliases)
        self._tag_name_lookup = {}
        try:
            all_tags = await self.stash.get_all_tags_with_aliases()
        except Exception:
            all_tags = await self.stash.get_all_tags()
        for t in all_tags:
            name = (t.get("name") or "").strip().lower()
            if name:
                self._tag_name_lookup[name] = str(t["id"])
            for alias in (t.get("aliases") or []):
                alias_lower = str(alias).strip().lower()
                if alias_lower:
                    self._tag_name_lookup.setdefault(alias_lower, str(t["id"]))

        # Studios: name + aliases
        self._studio_name_lookup = {}
        all_studios = await self.stash.get_all_studios()
        for s in all_studios:
            sid = str(s["id"])
            name = (s.get("name") or "").strip().lower()
            if name:
                self._studio_name_lookup[name] = sid
            for alias in (s.get("aliases") or []):
                alias_lower = alias.strip().lower()
                if alias_lower:
                    self._studio_name_lookup.setdefault(alias_lower, sid)

        logger.info(
            f"Name lookups built: {len(self._performer_name_lookup)} performer names, "
            f"{len(self._tag_name_lookup)} tags, {len(self._studio_name_lookup)} studios"
        )

    async def _process_endpoint(
        self, endpoint: str, api_key: str, incremental: bool,
        skip_local_ids: set[str] | None = None,
        endpoint_name: str | None = None,
    ) -> tuple[int, int]:
        """Store the current endpoint before processing for stash_id filtering."""
        self._current_endpoint = endpoint
        await self._build_name_lookups()
        return await super()._process_endpoint(endpoint, api_key, incremental, skip_local_ids, endpoint_name=endpoint_name)

    async def _get_local_entities(self, endpoint: str) -> list[dict]:
        return await self.stash.get_scenes_for_endpoint(endpoint)

    async def _get_upstream_entity(self, stashbox_client, stashbox_id: str) -> Optional[dict]:
        return await stashbox_client.get_scene(stashbox_id)

    def _build_local_data(self, entity: dict) -> dict:
        """Build comparable data from a local Stash scene.

        Maps local entity IDs to stashbox IDs via stash_ids for comparison
        with upstream data (which uses stashbox IDs natively).
        Filters stash_ids by the current endpoint to avoid cross-endpoint mismatches.
        """
        endpoint = self._current_endpoint
        performers = []
        for p in (entity.get("performers") or []):
            perf_stash_id = None
            for sid in (p.get("stash_ids") or []):
                if _endpoint_matches(sid.get("endpoint"), endpoint):
                    perf_stash_id = sid["stash_id"]
                    break
            if perf_stash_id:
                performers.append({
                    "id": perf_stash_id,
                    "name": p.get("name"),
                    "gender": p.get("gender"),
                    "as": None,
                })

        tags = []
        for t in (entity.get("tags") or []):
            for sid in (t.get("stash_ids") or []):
                if _endpoint_matches(sid.get("endpoint"), endpoint):
                    tags.append({"id": sid["stash_id"], "name": t.get("name")})
                    break

        studio = None
        local_studio = entity.get("studio")
        if local_studio:
            for sid in (local_studio.get("stash_ids") or []):
                if _endpoint_matches(sid.get("endpoint"), endpoint):
                    studio = {"id": sid["stash_id"], "name": local_studio.get("name")}
                    break

        urls = entity.get("urls") or []
        if isinstance(urls, str):
            urls = [urls] if urls else []

        return {
            "title": entity.get("title") or "",
            "date": entity.get("date") or "",
            "details": entity.get("details") or "",
            "director": entity.get("director") or "",
            "code": entity.get("code") or "",
            "urls": urls,
            "studio": studio,
            "performers": performers,
            "tags": tags,
            "_local_studio_id": str(local_studio.get("id")) if local_studio and local_studio.get("id") is not None else None,
            "_local_studio_name": local_studio.get("name") if local_studio else None,
            # Local scene membership (local IDs) used to suppress "add" suggestions
            # when the matched local entity is already on the scene.
            "_local_performer_ids": [str(p.get("id")) for p in (entity.get("performers") or []) if p.get("id") is not None],
            "_local_performer_stash_by_local_id": {
                str(p.get("id")): _normalize_stashbox_id(next((
                    sid.get("stash_id")
                    for sid in (p.get("stash_ids") or [])
                    if _endpoint_matches(sid.get("endpoint"), endpoint)
                ), None))
                for p in (entity.get("performers") or [])
                if p.get("id") is not None
            },
            "_local_tag_ids": [str(t.get("id")) for t in (entity.get("tags") or []) if t.get("id") is not None],
            "_local_tag_names": [(t.get("name") or "").strip().lower() for t in (entity.get("tags") or []) if (t.get("name") or "").strip()],
        }

    def _prune_added_changes_already_present(
        self,
        local_data: dict,
        performer_changes: Optional[dict],
        tag_changes: Optional[dict],
    ) -> tuple[dict, dict]:
        """Remove add-suggestions when the matched local entity is already on the scene."""
        local_performer_ids = set(local_data.get("_local_performer_ids") or [])
        local_performer_stash_by_local_id = local_data.get("_local_performer_stash_by_local_id") or {}
        local_tag_ids = set(local_data.get("_local_tag_ids") or [])
        local_tag_names = set(local_data.get("_local_tag_names") or [])

        performer_changes = performer_changes or {"added": [], "removed": [], "alias_changed": []}
        tag_changes = tag_changes or {"added": [], "removed": []}
        removed_performer_stash_ids = {
            _normalize_stashbox_id(p.get("id"))
            for p in performer_changes.get("removed", [])
            if _normalize_stashbox_id(p.get("id"))
        }

        pruned_added_performers = []
        for perf in performer_changes.get("added", []):
            match_id = None
            if self._performer_name_lookup:
                match_id = self._performer_name_lookup.get((perf.get("name") or "").strip().lower())
                if not match_id:
                    for alias in (perf.get("aliases") or []):
                        alias_key = str(alias).strip().lower()
                        if not alias_key:
                            continue
                        match_id = self._performer_name_lookup.get(alias_key)
                        if match_id:
                            break
            if match_id and str(match_id) in local_performer_ids:
                # Keep replacement additions when the matched local performer is also
                # being removed by stashbox ID in this same diff (merge/relink case).
                matched_local_stash_id = _normalize_stashbox_id(
                    local_performer_stash_by_local_id.get(str(match_id))
                )
                if matched_local_stash_id not in removed_performer_stash_ids:
                    continue
            pruned_added_performers.append(perf)

        pruned_added_tags = []
        for tag in tag_changes.get("added", []):
            tag_name = (tag.get("name") or "").strip().lower()
            if tag_name and tag_name in local_tag_names:
                continue
            match_id = None
            if self._tag_name_lookup:
                match_id = self._tag_name_lookup.get(tag_name)
            if match_id and str(match_id) in local_tag_ids:
                continue
            pruned_added_tags.append(tag)

        return (
            {
                "added": pruned_added_performers,
                "removed": performer_changes.get("removed", []),
                # Scene-level performer aliases are intentionally ignored.
                "alias_changed": [],
            },
            {
                "added": pruned_added_tags,
                "removed": tag_changes.get("removed", []),
            },
        )

    def _prune_studio_change_already_present(self, local_data: dict, studio_change: Optional[dict]):
        """Remove studio-change suggestion when the same local studio is already assigned."""
        if not studio_change:
            return studio_change

        upstream_studio = studio_change.get("upstream")
        if not upstream_studio:
            return studio_change

        local_scene_studio_id = str(local_data.get("_local_studio_id")) if local_data.get("_local_studio_id") is not None else None
        local_scene_studio_name = _normalize_name(local_data.get("_local_studio_name"))
        if not local_scene_studio_id and not local_scene_studio_name:
            return studio_change

        upstream_name = _normalize_name(upstream_studio.get("name"))

        # Most reliable fallback for this edge case: if the scene already has a
        # studio with the same name, don't suggest setting it again.
        if local_scene_studio_name and upstream_name and local_scene_studio_name == upstream_name:
            return None

        # If the upstream studio resolves to the same local studio ID already on
        # the scene, no studio update is required.
        if self._studio_name_lookup and local_scene_studio_id and upstream_name:
            matched_local_id = self._studio_name_lookup.get(upstream_name)
            if not matched_local_id:
                for alias in (upstream_studio.get("aliases") or []):
                    matched_local_id = self._studio_name_lookup.get(_normalize_name(alias))
                    if matched_local_id:
                        break
            if matched_local_id and str(matched_local_id) == local_scene_studio_id:
                return None

        return studio_change

    def _normalize_upstream(self, raw_data: dict) -> dict:
        return normalize_upstream_scene(raw_data)

    def _get_default_fields(self) -> set[str]:
        return DEFAULT_SCENE_FIELDS

    def _get_field_labels(self) -> dict[str, str]:
        return SCENE_FIELD_LABELS

    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        """Override: returns scene diff dict if changes exist, empty list if not.

        The base class checks `if not changes:` - so we return [] (falsy)
        when there are no changes, and the full dict (truthy) when there are.
        The _build_recommendation_details override handles the dict format.
        """
        result = diff_scene_fields(local_data, upstream_data, snapshot, enabled_fields)
        result["performer_changes"] = self._filter_performer_changes_by_gender(result.get("performer_changes"))
        result["performer_changes"], result["tag_changes"] = self._prune_added_changes_already_present(
            local_data,
            result.get("performer_changes"),
            result.get("tag_changes"),
        )
        result["studio_change"] = self._prune_studio_change_already_present(
            local_data,
            result.get("studio_change"),
        )
        if not _has_scene_changes(result):
            return []
        return result

    def _get_selected_performer_genders(self) -> set[str]:
        """Resolve enabled performer genders from settings."""
        selected: set[str] = set()
        for canonical, setting_key in GENDER_SETTING_KEY_BY_CANONICAL.items():
            try:
                enabled = bool(get_setting(setting_key))
            except Exception:
                # Settings system unavailable in isolated tests: keep defaults enabled.
                enabled = True
            if enabled:
                selected.add(canonical)
        return selected

    def _performer_gender_is_selected(self, performer: dict, selected: set[str]) -> bool:
        """Return True when performer gender is included by user selection."""
        gender = _normalize_performer_gender(performer.get("gender"))
        if not gender:
            gender = "UNKNOWN"
        if gender not in GENDER_SETTING_KEY_BY_CANONICAL:
            gender = "UNKNOWN"
        return gender in selected

    def _filter_performer_changes_by_gender(self, performer_changes: Optional[dict]) -> dict:
        """Filter performer added/removed changes by selected genders.

        Note: Alias differences are intentionally ignored for scene changes.
        """
        if not performer_changes:
            return {"added": [], "removed": [], "alias_changed": []}

        selected = self._get_selected_performer_genders()
        return {
            "added": [
                p for p in performer_changes.get("added", [])
                if self._performer_gender_is_selected(p, selected)
            ],
            "removed": [
                p for p in performer_changes.get("removed", [])
                if self._performer_gender_is_selected(p, selected)
            ],
            "alias_changed": [],
        }

    def _build_recommendation_details(
        self,
        endpoint: str,
        endpoint_name: str,
        stash_box_id: str,
        local_entity: dict,
        updated_at: Optional[str],
        changes,
    ) -> dict:
        """Build scene-specific recommendation details.

        The `changes` param is a dict from diff_scene_fields (not a list like other entities).
        """
        # Extract current local entity IDs so the UI can merge (not replace) on apply
        current_performer_ids = [
            str(p["id"]) for p in (local_entity.get("performers") or [])
        ]
        current_tag_ids = [
            str(t["id"]) for t in (local_entity.get("tags") or [])
        ]
        current_performers = [
            {"id": str(p["id"]), "name": p.get("name", "")}
            for p in (local_entity.get("performers") or [])
        ]
        current_tags = [
            {"id": str(t["id"]), "name": t.get("name", "")}
            for t in (local_entity.get("tags") or [])
        ]
        current_studio_id = None
        current_studio = None
        local_studio = local_entity.get("studio")
        if local_studio:
            current_studio_id = str(local_studio["id"])
            current_studio = {"id": current_studio_id, "name": local_studio.get("name", "")}

        details = {
            "endpoint": endpoint,
            "endpoint_name": endpoint_name,
            "stash_box_id": stash_box_id,
            "scene_id": str(local_entity["id"]),
            "scene_name": local_entity.get("title", ""),
            "upstream_updated_at": updated_at,
            "current_performer_ids": current_performer_ids,
            "current_tag_ids": current_tag_ids,
            "current_studio_id": current_studio_id,
            "current_performers": current_performers,
            "current_tags": current_tags,
            "current_studio": current_studio,
        }
        details["changes"] = changes.get("changes", [])
        details["studio_change"] = changes.get("studio_change")
        details["performer_changes"] = changes.get("performer_changes")
        details["tag_changes"] = changes.get("tag_changes")

        # Enrich added entities with local matches for auto-linking
        pc = details.get("performer_changes") or {}
        if pc.get("added") and self._performer_name_lookup:
            for perf in pc["added"]:
                match_id = self._performer_name_lookup.get(
                    (perf.get("name") or "").strip().lower()
                )
                if not match_id:
                    for alias in (perf.get("aliases") or []):
                        alias_key = str(alias).strip().lower()
                        if not alias_key:
                            continue
                        match_id = self._performer_name_lookup.get(alias_key)
                        if match_id:
                            break
                if match_id:
                    perf["local_match"] = {"id": match_id}

        tc = details.get("tag_changes") or {}
        if tc.get("added") and self._tag_name_lookup:
            for tag in tc["added"]:
                name = (tag.get("name") or "").strip().lower()
                match_id = self._tag_name_lookup.get(name)
                if match_id:
                    tag["local_match"] = {"id": match_id}

        sc = details.get("studio_change")
        if sc and not sc.get("local") and current_studio:
            sc["local"] = current_studio
        if sc and sc.get("upstream") and self._studio_name_lookup:
            upstream_studio = sc["upstream"]
            match_id = self._studio_name_lookup.get(
                (upstream_studio.get("name") or "").strip().lower()
            )
            if not match_id:
                for alias in (upstream_studio.get("aliases") or []):
                    match_id = self._studio_name_lookup.get(alias.strip().lower())
                    if match_id:
                        break
            if match_id:
                upstream_studio["local_match"] = {"id": match_id}

        return details
