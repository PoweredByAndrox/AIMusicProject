# Task List

## Preference Model

- Add a user-scoped preference intensity metric that is distinct from the 1-5 track rating scale, so extreme negative tracks and extreme positive tracks can shape training more strongly than ordinary ratings.
- Improve whole-track and segment rating reconciliation so an explicit track rating updates segment-derived state, and segment feedback updates the computed whole-track score consistently across radio, recommendations, and other screens.
- Improve recommendation quality by weighting the richer user-scoped signals more precisely during model training and recommendation generation.

## Segment Analysis

- Improve segment detection fallback for pathological tracks that collapse into a single `INTRO` segment across the full track.
- Add a concrete regression case for `Hit The Dirt (Preview) - The Vision ft. MC DL` and similar tracks so segment-analysis changes can be verified against known bad examples.
