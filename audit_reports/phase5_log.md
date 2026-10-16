# Phase 5 log: running notes

Pre-registration: `phase5_preregistration.md` (main text committed in `fc0c7c8` before any Phase 4 result; Addendum A for P5a committed in `4025af4` before any flow data was downloaded or used).

## Step 0: raw-cut statistics (reporting only, no selection) — Umang's instruction after Phase 4

- `p5_raw_stats.py`, Phase 4 statistics on the raw cut (CLEANBLEED):
  - P vs naive: significant at every horizon (DM p <= 0.0001; BH 13 / 14 / 14 MSAs).
  - P vs FA-ETS-d: 4wk +0.240 (p = 0.003, FA-ETS-d better in 4 MSAs after BH); 8wk +0.055 (p = 0.33); 13wk +0.025 (p = 0.75). Not equivalent within 2% at any horizon (CIs too wide at 8 / 13).
  - P vs seasonal channel (Naive2): 4wk +0.015 (p = 0.83); 8wk -0.249 (p = 0.003); 13wk -0.300 (p = 0.011).
  - R1c vs Naive2: no significant difference at any horizon. Holt vs naive: no significant difference at any horizon.
  - 90% MCS: 4wk {FA-ETS-d, C2, P, Naive2} (the last three at p = 0.100); 8wk {FA-ETS-d, P, C2}; 13wk {FA-ETS-d, P, C2, R1c, Naive2}.
