# Scene Tagger

This page documents the exact Scene Stash-Box Tagger matching behavior currently implemented.

## Prerequisites

- **At least one Stash-Box endpoint** configured in Stash (**Settings > Metadata Providers**)
- Local scenes must have file fingerprints (MD5/OSHASH/PHASH) available in Stash

## Matching Rules In Effect

### 1. Run scope and incremental behavior

- The analyzer runs once per configured Stash-Box endpoint.
- It reads scenes through `get_scenes_with_fingerprints(...)` in pages of `100`.
- Incremental mode uses a per-endpoint watermark key:
  - `scene_fp_match_{endpoint}`
  - filter: `scene.updated_at > last_stash_updated_at`
- Full scans ignore the watermark.
- User-triggered queue runs of `scene_fingerprint_match` default to full scan (`cursor="__full__"`), unless explicitly overridden.

### 2. Local scene eligibility (strict filter)

A local scene is considered for matching only if all of the following are true:

- It has **no stash IDs at all** (`scene.stash_ids` must be empty).
- It has at least one file fingerprint across its files.

Notes:

- If a scene has any stash ID (even from another endpoint), it is skipped.
- Local duration used for scoring is the first available file duration on the scene.

### 3. Endpoint lookup

- Candidate scenes are queried in batches of `40` (`BATCH_SIZE = 40`).
- Query used: `findScenesBySceneFingerprints`.
- Each local scene in the batch returns zero or more remote scene matches for that endpoint.

### 4. Per-match scoring and confidence

For each local scene ↔ stash-box scene match:

- `target_id` is composite: `"{local_scene_id}|{endpoint}|{stashbox_scene_id}"`.
- Match fingerprints are counted by hash overlap:
  - local hashes are collected into a set
  - remote fingerprint entries are counted if `remote.hash` exists in that set

Score fields:

- `match_count = len(matching_fingerprints)`
- `match_percentage = (match_count / total_local_fingerprints) * 100` (rounded to 1 decimal)
- `has_exact_hash = True` if any matching fingerprint algorithm is `MD5` or `OSHASH`
- `total_submissions = sum(fp.submissions for matching_fingerprints)`
- Duration agreement:
  - compare local duration with each matching fingerprint duration
  - `duration_diff = average(abs(local - remote))`
  - `duration_agreement = (duration_diff <= 5.0s)`
  - if no comparable durations exist, `duration_agreement = True` and `duration_diff = null`

Stored recommendation confidence:

- `confidence = match_percentage / 100.0` (0.0 to 1.0 scale in DB/API)

High-confidence flag:

- `high_confidence = (not ambiguous) AND (match_count >= min_count) AND (match_percentage >= min_percentage)`
- A scene is considered ambiguous when it has more than one match from the endpoint (`len(matches) > 1`).
- Defaults:
  - `scene_fp_min_count = 2`
  - `scene_fp_min_percentage = 66`

### 5. Dismiss and accept behavior

- Dismiss checks are exact on composite `target_id`, so dismissal is per pair (local scene + endpoint + remote scene).
- Accepting one match:
  - appends that stash ID to local scene (if not already linked)
  - resolves that recommendation as accepted
  - auto-dismisses all other pending `scene_fingerprint_match` recommendations for the same local scene ID
- "Accept all" processes only pending recommendations with `details.high_confidence == true` (optional endpoint filter supported).

### 6. Result ordering in lists

`scene_fingerprint_match` recommendations are returned sorted by:

1. `confidence DESC`
2. `created_at DESC`
3. `id DESC`

This ordering applies consistently across pending/resolved/dismissed status views.
