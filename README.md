# callosal-dims

What does the corpus callosum carry? Separating interhemispheric communication from common drive
with unilateral silencing in mouse ALM, toward the minimal specification of a callosal BMI.

## Documents (Japanese)

- [docs/01_mouse_callosal_dims.md](docs/01_mouse_callosal_dims.md) — background, questions, phases and results of this project
- Follow-up master's research plan (split-brain world models): [Nzm3915/split-brain-world-models](https://github.com/Nzm3915/split-brain-world-models)

Data: [DANDI 000894](https://dandiarchive.org/dandiset/000894) — simultaneous bilateral ALM recordings
with unilateral / bilateral photoinhibition (Chen, Kang, Lindsey, Druckmann & Li, *Cell* 2021).

## Phase 0 — recover hemisphere labels and photostimulation codes

The NWB files list a single 64-channel electrode table with location "Left ALM, Right ALM", and the
trial table codes photoinhibition only as `PhotostimulationType` 0–3. Phase 0 recovers both from the
data: unilateral photoinhibition (VGAT-ChR2 / PV-ReaChR) silences pyramidal neurons only in the
illuminated hemisphere, so the code that silences a unit tells which hemisphere it belongs to.

```
pip install -r requirements.txt
python phase0/phase0_labels.py results/sessions      # streams spike times only (no 270 GB download)
python phase0/phase0_classify.py results/sessions/ALL.csv results
```

Window: 0.10–0.70 s after pole-out (= delay onset). The laser runs from pole-out for 0.8 s, the
last 0.2 s being a linear ramp-down. Pole-out is 1.87 s after trial start in 67 sessions and 2.87 s
in the other 66, so the window is aligned per trial.

### Results (133 sessions, 39 mice, 19,443 units)

![phase0](results/fig_phase0_labels.png)

| Finding | Evidence |
| --- | --- |
| Type 3 = bilateral photoinhibition | Laser peak power 13.7 mW vs 4.43 mW for types 1/2 (8-site galvo scanning); 93.9% of pyramidal units drop below 30% of control rate |
| Types 1 and 2 = the two unilateral conditions | 11,933 of 12,999 pyramidal units (≥1 Hz) are silenced (<20% of control) by exactly one of them, at least 4x more than by the other; 5,929 vs 6,004 |
| Units can be assigned to a hemisphere | Median rate ratio: code-1 units 0.00 under type 1 vs 0.88 under type 2; code-2 units 0.88 vs 0.005 |
| The electrode index does **not** identify the hemisphere | Electrode indices of the two groups overlap in every session (0 of 133 cleanly separated); the NWB electrode table has 64 rows for two 64-ch probes |
| Fast-spiking units behave as expected for activated inhibitory cells | Median rate ratio 3.05 under bilateral photoinhibition |
| Enough units per hemisphere for population analyses | Median 42–43 labeled units per hemisphere per session; 116 of 133 sessions have ≥20 in each |

Which code is *left*: the laser-site metadata lists the coordinates in the order (AP 2.5, ML −1.5),
(AP 2.5, ML +1.5) and the location "left ALM right ALM", which suggests type 1 = left ALM and
type 2 = right ALM. This is not yet confirmed independently. The planned analyses are symmetric
in the two hemispheres, so they do not depend on it.

## Phase 1 — reproduce Chen et al. 2021 Fig. 1, and look at error trials

```
python phase1/cache_counts.py CACHE_DIR                      # 50-ms spike counts aligned to the go cue (~110 MB)
python phase1/phase1_cd.py CACHE_DIR results/units_labeled.csv results/phase1
python phase1/phase1_plot.py results/phase1
```

Choice decoder (CD) per hemisphere = mean over the delay of (mean rate on "lick right" − "lick left" correct
trials), as in Chen et al.; 400-ms windows; 2-fold cross-validation so every correct trial is projected with a
CD that did not see it. Endpoint = last 400 ms before the go cue. Control trials only (no photostimulation,
no early lick, normal 1.7-s delay). Only Phase-0-labeled pyramidal units (median 33 in the smaller hemisphere;
Chen et al. averaged 71 neurons per hemisphere).

![phase1](results/phase1/fig_phase1_cd.png)

| Finding | Value (133 sessions) |
| --- | --- |
| Endpoints correlate between hemispheres within a trial type (Fig. 1G) | Spearman ρ 0.26 ("lick right"), 0.27 ("lick left"), 0.55 (both types); shuffled ≈ 0; within-type vs shuffled p = 7e-20 |
| The two hemispheres predict the same choice (Fig. 1H) | 74% of held-out correct trials; each hemisphere predicts the lick on 80–81% |
| The shortfall versus Chen et al. is explained by neuron count | ρ and agreement rise with the number of labeled units (Spearman 0.54, p = 3e-11); sessions with ≥ 60 units in both hemispheres: agreement 83%, ≥ 80 units: 89% |
| The correlation rises over the trial (Fig. S1H) | within-type ρ 0.14 before the sample epoch → 0.28 in the late delay → 0.07 after the go cue |
| The pre-sample correlation is not slow drift across the session | removing a ±10-trial running mean leaves it at 0.13 |
| **New:** on error trials the hemispheres make the same mistake more often than chance | agreement 61% vs 53% expected if independent given the trial type (p = 8e-12, 5,833 error trials); both predict the wrong (actual) lick on 38% of errors, both the instructed side on 23% |

Reading: about half of the late-delay correlation (ρ ≈ 0.14 of 0.28) is already present before the stimulus,
i.e. it is shared fluctuation that is not about the choice — a candidate for common input such as arousal
(McDonnell et al. 2026). Separating that part from what one hemisphere actually sends the other is Phase 2.

Next (Phase 2): correlated subspace (reduced-rank regression, pCCA-FA) versus receiving subspace (effect of
unilateral photoinhibition on the other hemisphere), and the angles between them.
