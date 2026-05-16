"""
Upstream Performer Changes Analyzer

Detects changes in stash-box linked performers by comparing upstream
data against local Stash data using 3-way diffing with stored snapshots.
"""

import logging
from typing import Optional

from .base_upstream import BaseUpstreamAnalyzer
from stashbox_client import StashBoxClient
from upstream_field_mapper import (
    normalize_upstream_performer,
    diff_performer_fields,
    DEFAULT_PERFORMER_FIELDS,
    FIELD_LABELS,
)

logger = logging.getLogger(__name__)


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


def _build_relinked_stash_ids(
    current_stash_ids: list[dict],
    endpoint: str,
    new_stashbox_id: str,
) -> list[dict]:
    """Replace endpoint stash_id mapping with merged target stash_id."""
    relinked: list[dict] = []
    seen: set[tuple[str, str]] = set()
    replaced = False

    for sid in current_stash_ids or []:
        sid_endpoint = sid.get("endpoint")
        sid_stash_id = sid.get("stash_id")
        if not sid_endpoint or not sid_stash_id:
            continue

        if _endpoint_matches(sid_endpoint, endpoint):
            if not replaced:
                key = (sid_endpoint, new_stashbox_id)
                if key not in seen:
                    relinked.append({"endpoint": sid_endpoint, "stash_id": new_stashbox_id})
                    seen.add(key)
                replaced = True
            continue

        key = (sid_endpoint, sid_stash_id)
        if key in seen:
            continue
        relinked.append({"endpoint": sid_endpoint, "stash_id": sid_stash_id})
        seen.add(key)

    if not replaced:
        key = (endpoint, new_stashbox_id)
        if key not in seen:
            relinked.append({"endpoint": endpoint, "stash_id": new_stashbox_id})

    return relinked


def _build_local_performer_data(performer: dict) -> dict:
    """Extract comparable field values from a local Stash performer."""
    from upstream_field_mapper import parse_measurements, parse_career_length

    measurements = parse_measurements(performer.get("measurements"))
    career = parse_career_length(performer.get("career_length"))

    return {
        "name": performer.get("name"),
        "disambiguation": performer.get("disambiguation") or "",
        "aliases": performer.get("alias_list") or [],
        "gender": performer.get("gender"),
        "birthdate": performer.get("birthdate"),
        "death_date": performer.get("death_date"),
        "ethnicity": performer.get("ethnicity"),
        "country": performer.get("country"),
        "eye_color": performer.get("eye_color"),
        "hair_color": performer.get("hair_color"),
        "height": performer.get("height_cm"),
        "cup_size": measurements["cup_size"],
        "band_size": measurements["band_size"],
        "waist_size": measurements["waist_size"],
        "hip_size": measurements["hip_size"],
        "breast_type": performer.get("fake_tits"),
        "tattoos": performer.get("tattoos") or "",
        "piercings": performer.get("piercings") or "",
        "career_start_year": career["career_start_year"],
        "career_end_year": career["career_end_year"],
        "urls": performer.get("urls") or [],
    }


class UpstreamPerformerAnalyzer(BaseUpstreamAnalyzer):
    """
    Detects upstream changes in stash-box linked performers.

    Compares performer data from configured stash-box endpoints against
    local Stash data, using stored snapshots for 3-way diffing.
    """

    type = "upstream_performer_changes"
    logic_version = 5  # v5: surface merged/deleted upstream performers with relink metadata

    @property
    def entity_type(self) -> str:
        return "performer"

    async def _get_local_entities(self, endpoint: str) -> list[dict]:
        return await self.stash.get_performers_for_endpoint(endpoint)

    async def _get_upstream_entity(self, stashbox_client: StashBoxClient, stashbox_id: str) -> Optional[dict]:
        performer = await stashbox_client.get_performer(stashbox_id)
        if not performer:
            return None

        merged_into_id = performer.get("merged_into_id")
        if merged_into_id:
            try:
                merged_target = await stashbox_client.get_performer(merged_into_id)
            except Exception as e:
                logger.warning(
                    "Failed to fetch merged target performer %s for %s: %s",
                    merged_into_id,
                    stashbox_id,
                    e,
                )
                merged_target = None
            if merged_target:
                performer["_merged_target"] = merged_target

        return performer

    def _build_local_data(self, entity: dict) -> dict:
        return _build_local_performer_data(entity)

    def _normalize_upstream(self, raw_data: dict) -> dict:
        source = raw_data
        status = "active"

        if raw_data.get("merged_into_id"):
            status = "merged"
            source = raw_data.get("_merged_target") or raw_data
        elif raw_data.get("deleted"):
            status = "deleted"

        normalized = normalize_upstream_performer(source)
        normalized["_upstream_status"] = status
        normalized["_source_stashbox_id"] = raw_data.get("id")
        normalized["_merged_into_id"] = raw_data.get("merged_into_id")
        normalized["_merged_target_id"] = (raw_data.get("_merged_target") or {}).get("id")
        normalized["_merged_target_name"] = (raw_data.get("_merged_target") or {}).get("name")
        return normalized

    def _is_upstream_deleted(self, upstream: dict) -> bool:
        """Performer sync surfaces deleted/merged upstream entities as recommendations."""
        return False

    def _get_upstream_updated_at(self, upstream: dict) -> Optional[str]:
        """Use merged target updated timestamp when available for better watermarking."""
        merged_target = upstream.get("_merged_target")
        if merged_target and merged_target.get("updated"):
            return merged_target.get("updated")
        return upstream.get("updated")

    def _get_default_fields(self) -> set[str]:
        return DEFAULT_PERFORMER_FIELDS

    def _get_field_labels(self) -> dict[str, str]:
        return FIELD_LABELS

    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        changes = diff_performer_fields(local_data, upstream_data, snapshot, enabled_fields)

        upstream_status = upstream_data.get("_upstream_status") or "active"
        if upstream_status == "deleted":
            changes.insert(0, {
                "field": "_upstream_status",
                "field_label": "Upstream Status",
                "local_value": "ACTIVE",
                "upstream_value": "DELETED",
                "previous_upstream_value": "ACTIVE",
                "merge_type": "readonly",
            })
        elif upstream_status == "merged":
            source_stashbox_id = upstream_data.get("_source_stashbox_id")
            merged_stashbox_id = (
                upstream_data.get("_merged_target_id")
                or upstream_data.get("_merged_into_id")
            )
            if (
                source_stashbox_id
                and merged_stashbox_id
                and _normalize_stashbox_id(source_stashbox_id) != _normalize_stashbox_id(merged_stashbox_id)
            ):
                changes.insert(0, {
                    "field": "_stashbox_id",
                    "field_label": "StashBox Performer ID",
                    "local_value": source_stashbox_id,
                    "upstream_value": merged_stashbox_id,
                    "previous_upstream_value": source_stashbox_id,
                    "merge_type": "readonly",
                })

        return changes

    def _build_recommendation_details(
        self,
        endpoint: str,
        endpoint_name: str,
        stash_box_id: str,
        local_entity: dict,
        updated_at: Optional[str],
        changes: list[dict],
    ) -> dict:
        details = super()._build_recommendation_details(
            endpoint=endpoint,
            endpoint_name=endpoint_name,
            stash_box_id=stash_box_id,
            local_entity=local_entity,
            updated_at=updated_at,
            changes=changes,
        )
        # Add performer-specific fields
        details["performer_image_path"] = local_entity.get("image_path")
        details["performer_disambiguation"] = local_entity.get("disambiguation") or ""

        upstream_status = "active"
        merged_target_stashbox_id = None
        for change in changes:
            if change.get("field") == "_upstream_status" and str(change.get("upstream_value", "")).upper() == "DELETED":
                upstream_status = "deleted"
            if change.get("field") == "_stashbox_id":
                upstream_status = "merged"
                merged_target_stashbox_id = change.get("upstream_value")

        details["upstream_status"] = upstream_status
        if merged_target_stashbox_id:
            details["merged_target_stashbox_id"] = merged_target_stashbox_id
            details["relink"] = {
                "endpoint": endpoint,
                "old_stashbox_id": stash_box_id,
                "new_stashbox_id": merged_target_stashbox_id,
                "stash_ids_after_relink": _build_relinked_stash_ids(
                    local_entity.get("stash_ids") or [],
                    endpoint=endpoint,
                    new_stashbox_id=merged_target_stashbox_id,
                ),
            }

        return details
