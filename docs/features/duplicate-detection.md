# Duplicate Detection

This page documents the exact duplicate scene detection behavior currently implemented.

## Prerequisites

- Stash scene metadata (studio, performers, date, stash IDs) available through Stash GraphQL
- Perceptual hashes (phash) available on scene files for best coverage
- Face fingerprints are optional but improve non-authoritative matching

## Rules In Effect

### 1. Run behavior and cleanup

- The analyzer currently runs as a full scan in practice (incremental mode is not used by the algorithm).
- At run start it deletes all **pending** `duplicate_scenes` recommendations from previous runs.
  - Purpose: avoid stale pending rows and allow re-insertion with fresh scoring.
  - Dismissed targets remain dismissed and continue to block recreation.
- It clears the full `duplicate_candidates` table before generating candidates.
  - Reason: uniqueness is on `(scene_a_id, scene_b_id)` and old rows would block new inserts.

### 2. Candidate generation phase (three sources)

Candidate pairs are generated as canonical ordered pairs `(min(scene_id), max(scene_id))` and inserted with `INSERT OR IGNORE`.

Source A: Shared stash-box ID on same endpoint

- From scenes returned by `get_scenes_with_fingerprints(...)`.
- If two scenes share `(endpoint, stash_id)`, they become candidates.

Source B: Phash Hamming distance

- Uses the first available `phash` fingerprint per scene.
- Generates pairs where `HammingDistance(phash_a, phash_b) <= 10`.
- Distance values are kept in memory for scoring phase.

Source C: Metadata intersection

- From `get_scenes_for_fingerprinting(...)`.
- Index key is `(studio_id, performer_id)`.
- Scenes that share at least one such key become candidates.

### 3. Scoring phase entry criteria

For each candidate pair:

- Scene metadata is loaded as `SceneMetadata` (title, date, studio, performers, duration, stash IDs).
- Face fingerprints are loaded from DB only where `scene_fingerprints.fingerprint_status = 'complete'`.
- `calculate_duplicate_confidence(...)` is executed once per pair.
- A recommendation is created only when `confidence >= min_confidence`.
  - Default `min_confidence = 50.0`.

Stored recommendation identity:

- `target_id = "{scene_a_id}:{scene_b_id}"` (canonical pair order)

### 4. Signal scoring formulas

#### Tier 1: Authoritative stash-box match

- If scenes share identical stash-box ID on same endpoint:
  - confidence = `100.0`
  - no further scoring needed

#### Phash score (0 to 85)

- If distance is `None`: `0`
- If distance `> 10`: `0`
- Else: `85 - (distance * 6.5)`

#### Face signature score (0 to 75)

- Uses performer sets from fingerprints, excluding performer ID `"unknown"`.
- Requires at least one shared performer.
- `jaccard = |shared| / |union|`
- `base = jaccard * 60`
- `proportion_bonus = max(0, 15 * (1 - avg_shared_proportion_diff * 4))`
- `face_score = min(base + proportion_bonus, 75)`

#### Metadata score (0 to 60, clamped)

Primary positives:

- Same date + exact performer set: `+45`
- Same date + performer overlap >= 50%: `+35`
- Date within 7 days + exact performer set: `+35`
- Exact performer set: `+25`
- Same date (with performer data present): `+20`
- Performer overlap >= 50%: `+15`

Additional boost:

- Same studio: `+10`

Duration penalties (never positive):

- Duration ratio `< 15%`: `-20`
- Duration ratio `< 30%`: `-10`

#### Final confidence combination

- If `phash_distance <= 4`:
  - `confidence = phash + min(metadata * 0.15, 5) + min(face * 0.1, 5)`
- Else if `phash > 0`:
  - With corroboration (`max(metadata, face) > 0`):
    - `confidence = phash + max(metadata, face) * 0.4`
  - Without corroboration:
    - `confidence = phash * 0.6`
- Else (no phash signal):
  - `confidence = max(metadata, face) + min(metadata, face) * 0.3`

Final cap:

- Non-authoritative matches are capped at `95.0`.

### 5. Result ordering in lists

`duplicate_scenes` recommendations are returned sorted by:

1. `confidence DESC`
2. `created_at DESC`
3. `id DESC`

This ordering applies consistently across pending/resolved/dismissed status views.
