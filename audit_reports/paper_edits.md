# FACTS-MAS paper: every edit, in paper order

For: Umang, applying the edits in Overleaf. Replaces `paper_edits_part1.md`.

## How to use this file

Work from top to bottom. The steps follow the order of your `.tex` file, so you never jump back.

Every step has the same four parts:

- **Where**: the section and paragraph.
- **Search for**: a short phrase that sits on **one line** of your `.tex`. Paste it into Overleaf's search (Ctrl+F) to land on the spot.
- **Do**: one of **REPLACE**, **INSERT AFTER**, or **DELETE**, and exactly what is covered.
- **New text**: LaTeX to paste, written in the paper's own style.

When a step says "REPLACE from X through Y", select from the first line X down to and including the last line Y, delete it, and paste the new text.

After the last step, run the leftover check in Step 60 to catch any old number still in the paper.

### Numbers used in the new text

- **Backtest:** six folds, test origins, targets dated before January 1, 2026 (2,610 / 2,565 / 2,490 cells at 4 / 8 / 13 weeks).
- **Holdout:** origins February 28 to September 12, 2026 (29 / 25 / 20 weeks).
- **Smoothed cut is primary** unless a step says "unsmoothed".
- All numbers come from `audit_reports/` (phase reports and `results_tables/`).

### Labels added, removed or kept

| Label | Status |
|---|---|
| `sec:holdout` | **new** (Step 46) |
| `tab:holdout` | **new** (Step 46) |
| `tab:holdout_tests` | **new** (Step 46) |
| `tab:ui_results` | **new** (Step 51) |
| `tab:event_scale_analysis` | **removed** (Step 55) |
| `tab:fusion_modes` | **removed** (Step 56) |
| `tab:granular_regression` | **removed** (Step 56) |
| `tab:housing_primary`, `tab:housing_strong_baselines` | **kept**, contents replaced |

## Quick map of all steps

| Step | Section | Action |
|---|---|---|
| 1 | Abstract | Replace whole abstract |
| 2-3 | 1 Introduction | Fix "lags calibrated"; replace results paragraph |
| 4-7 | 1 Contributions, roadmap | Update items 2-4, add item 5, fix roadmap |
| 8-11 | 2 Related Work | Name temporal models; add one sentence; rewrite NEXUS paragraph |
| 12-20 | 3 Framework | Weights vs calibration; Figure 1 caption; tau; per-fold weights; sufficiency rule; calibration disclosure; attribution |
| 21-27 | 4 Case Studies, 4.1 protocol | Two cases; protocol disclosures; statistics; ablation; weighting rules; baselines |
| 28-44 | 4.2.1 Domain instantiation | Data; channel table; each channel paragraph; event calibration table; folds; regimes; fusion configurations |
| 45-48 | 4.2.2 Forecasting results | New Table 5; new Table 6; per-regime paragraph; new holdout subsection; limits |
| 49-51 | 4.3, 4.4 Energy, Unemployment | Energy marked planned; unemployment results added |
| 52-58 | 5 Discussion | Channel summary; macro table; event table; fusion analysis rewritten; full-window; implications; transfer hypotheses |
| 59 | 6 Conclusion | Replace results paragraph |
| 60 | Bibliography and leftover check | 10 new entries; search for stale numbers |

---

# ABSTRACT

## Step 1. Abstract (replace all of it)

**Search for:** `Real-world forecasting often requires integrating heterogeneous`

**Do:** REPLACE everything between `\noindent` and `\end{abstract}` (the single long paragraph).

**New text:**
```latex
Real-world forecasting often requires integrating heterogeneous evidence whose predictive value varies across horizons, operating regimes, and domains. We introduce FACTS-MAS, a domain-agnostic multi-agent forecasting framework that decomposes evidence into specialized channels for temporal dynamics, exogenous context, seasonality, structural information, and discrete events. Each agent generates an independently testable forecast through a common interface, while leakage-safe historical validation measures its predictive skill relative to a naive reference. These skill estimates are used to weight and combine agent forecasts, so that uninformative evidence is downweighted rather than forced to participate. A calibration procedure selects domain-specific constants on historical validation folds and may set a constant to zero when a nonzero effect is not supported. We evaluate FACTS-MAS on weekly for-sale housing inventory for 15 U.S.\ metropolitan areas at horizons of 4, 8, and 13 weeks. Relative to a last-value forecast, the frozen configuration reduces mean absolute percentage error by 50\%, 50\%, and 47\% on Zillow's smoothed inventory series and by 24\%, 35\%, and 37\% on the unsmoothed series, and it retains these gains on a pre-registered holdout of observations after February 21, 2026 that were not accessed until the configuration was frozen. It also outperforms the TimesFM foundation model. However, a factor-adjusted damped exponential smoothing model performs as well as the fused forecast, and two of the five housing channels calibrate to a null effect. The evidence therefore supports FACTS-MAS as a systematic protocol for testing, calibrating, and recomposing evidence channels, rather than as a source of accuracy beyond the strongest single model.
```

**Why:**
- The old abstract had no results.
- Its calibration sentence ("a calibration layer learns ... evidence mappings") overstated what calibration does.
- It now names the smoothed cut and gives the gain on both cuts, as agreed.

---

# 1 INTRODUCTION

## Step 2. Paragraph 3: constants sentence

**Search for:** `Domain-specific constants---lags, impact scales, and`

**Do:** REPLACE this sentence (it runs over three lines):
```latex
Domain-specific constants---lags, impact scales, and
durations---are chosen by the same validation test, and these constants can
be set to zero.
```

**New text:**
```latex
Domain-specific constants---event-impact scales, event durations, and
reversion caps---are chosen by the same validation test, and these constants
can be set to zero.
```

**Why:** the macro lags were fixed from the literature, not chosen by validation.

## Step 3. Paragraph 4: the results paragraph

**Search for:** `We evaluate FACTS-MAS on the forecast of weekly U.S.\ housing inventory, a setting in`

**Do:** REPLACE from that line through the line ending `assess the generalizability of the approach.`

**New text:**
```latex
We evaluate FACTS-MAS on the forecast of weekly U.S.\ housing inventory, a setting in
which all five candidate channels can be populated: recent inventory
dynamics, national financial conditions, annual listing seasonality,
metropolitan structure, and disaster declarations. The study covers 15
metropolitan statistical areas (MSAs) at forecast horizons of 4, 8, and 13
weeks. At each forecast origin, defined as the week in which a prediction is issued,
models are trained only on observations dated on or before that week; because
Zillow publishes each week's value one to two weeks later, we also report every
result with a two-week publication lag. Evaluation uses a six-fold expanding-window
backtest, with a 13-week embargo between the end of training and the start
of testing, followed by a single pre-registered holdout on data after
February 21, 2026 that was not accessed until the configuration was frozen.
On Zillow's smoothed inventory series, the frozen configuration reduces MAPE
relative to a naive last-value forecast from 4.194\% to 2.118\% at the 4-week
horizon, from 7.225\% to 3.622\% at 8 weeks, and from 10.814\% to 5.731\% at
13 weeks in the backtest, and from 3.122\% to 1.243\%, 5.445\% to 1.929\%, and
8.438\% to 3.045\% on the holdout. The reduction is smaller on the unsmoothed
series (24\% at 4 weeks in the backtest, against 50\% on the smoothed series).
The fused forecast is significantly more accurate than the TimesFM foundation
model \citep{das2024timesfm} in the backtest. On the holdout it has a lower
error than a chain-of-thought language-model forecast built on the NEXUS
prompt \citep{das2026nexus}, but the difference is not statistically
significant. It is not more accurate than a
factor-adjusted damped exponential smoothing model, with which it is
statistically equivalent at 8 weeks. We also transfer the method to weekly
unemployment insurance claims to assess the generalizability of the approach.
```

**Why:**
- The old numbers (2.243 / 5.038 / 7.020) came from a July run of an older system.
- The fold-1 NEXUS claim used that old system.
- "Trained only on observations available" ignored Zillow's publication delay.

## Step 4. Contribution 2: weights sentence

**Search for:** `depend on forecast horizon and operating regime when the validation`

**Do:** REPLACE these two lines:
```latex
    depend on forecast horizon and operating regime when the validation
    sample is sufficient to support that conditioning.
```

**New text:**
```latex
    depend on forecast horizon and operating regime when the validation
    sample is sufficient to support that conditioning; a regime is used only
    when it contains at least 13 distinct validation weeks, and otherwise the
    pooled weights apply.
```

**Why:** the original code had no sufficiency rule, so this states the rule that the frozen system actually uses.

## Step 5. Contribution 3: list of constants

**Search for:** `including lags, impact scales, and event durations. Candidate values are`

**Do:** REPLACE only the words `including lags, impact scales, and event durations.`

**New text:**
```latex
including event-impact scales, event durations, and reversion caps.
```

## Step 6. Contribution 4: evaluation item (and a new item 5)

**Search for:** `\item We evaluate the method on weekly for-sale housing inventory for`

**Do:** REPLACE from that `\item` line through the line ending `effect is weak or unsupported.`. The `\end{enumerate}` that follows stays.

**New text:**
```latex
    \item We evaluate the method on weekly for-sale housing inventory for
    15 U.S.\ metropolitan areas at forecast horizons of 4, 8, and 13 weeks,
    on both Zillow's smoothed and unsmoothed series. The analysis compares
    accuracy with naive, seasonal-naive, and seasonally adjusted naive
    references, with exponential-smoothing and Theta models, with the
    TimesFM foundation model, and with a chain-of-thought language-model
    forecast, using panel Diebold--Mariano tests and block-bootstrap
    confidence intervals. It distinguishes evidence channels that add
    incremental skill from channels whose quantitative effect is weak or
    unsupported.

    \item We report a pre-registered evaluation: the configuration, the
    hypotheses, and the statistical tests were fixed and committed before a
    holdout period of 29 weekly origins was accessed, and every result on
    that holdout is reported, including those that do not favor the fused
    forecast.
```

## Step 7. Roadmap sentence

**Search for:** `Section~\ref{sec:discussion} discusses the housing findings`

**Do:** REPLACE these three lines:
```latex
Section~\ref{sec:discussion} discusses the housing findings
and describes transfer of the method to other forecasting tasks, including
unemployment.
```

**New text:**
```latex
Section~\ref{sec:discussion} discusses the housing findings, including the
holdout results, and the transfer of the method to weekly unemployment
insurance claims.
```

---

# 2 RELATED WORK

## Step 8. Section 2.2, last sentence: name the temporal models

**Search for:** `uses Holt--Winters on the temporal channel and TimesFM~2.5, a later checkpoint`

**Do:** REPLACE the sentence that starts `The housing study` (end of the previous line) through `as a numerical baseline.`:
```latex
The housing study
uses Holt--Winters on the temporal channel and TimesFM~2.5, a later checkpoint
of the TimesFM line, as a numerical baseline.
```

**New text:**
```latex
The housing study uses Holt's linear-trend exponential smoothing on the
temporal channel at the 4-week horizon and a factor-adjusted damped-trend
model \citep{gardner1985damped} at 8 and 13 weeks, and it uses TimesFM~2.5,
a later checkpoint of the TimesFM line, as a numerical baseline.
```

## Step 9. Section 2.3: one sentence added after Timmermann

**Search for:** `\citet{timmermann2006combinations} reviews when such combination improves on`

**Do:** INSERT AFTER the next line, which ends `the best individual forecast.`

**New text:**
```latex
That question is central here: a combination of channels is only useful if
it improves on the strongest single model that could have been selected on
the same validation data.
```

## Step 10. Section 2.4: the NEXUS paragraph

**Search for:** `The immediate precursor of our MAS approach is NEXUS`

**Do:** REPLACE the whole paragraph, from that line through the line ending `adapts to the channels that have recently proved useful.`

**New text:**
```latex
The immediate precursor of our MAS approach is NEXUS \citep{das2026nexus}, which forecasts from
mixed structured and unstructured evidence by passing that evidence through
multi-agent language-model reasoning and then issuing the forecast.
Efficiency extensions of that design, such as early-exit gating and batched
reasoning, retain language-model reasoning on the main forecasting path; they
are not part of FACTS-MAS and are not evaluated here. FACTS-MAS assigns the
same evidence to specialized channels and obtains an independent numerical
forecast from each channel. The fusion weights are estimated from the
channels' historical skill relative to a simple reference on a validation
window that ends at the training cutoff, and they are re-estimated whenever
the training window is extended. Those weights may also depend on the
forecast horizon and the operating regime when the validation sample is
sufficient.
```

**Why:**
- The four NEXUS modifications were removed from this paper.
- The weights are not updated at every origin. The code estimates them once per fold (new mismatch item B17).

## Step 11. Section 2.4: last paragraph

**Search for:** `The housing experiments in Section~\ref{sec:housing} compare FACTS-MAS with`

**Do:** REPLACE these two lines:
```latex
The housing experiments in Section~\ref{sec:housing} compare FACTS-MAS with
a NEXUS chain-of-thought baseline and with a time-series foundation model.
```

**New text:**
```latex
The housing experiments in Section~\ref{sec:housing} compare FACTS-MAS with
a chain-of-thought language-model forecast that uses the NEXUS prompt, with a
time-series foundation model, and with strong statistical references.
```

---

# 3 FRAMEWORK

## Step 12. Opening, the paragraph after equation (3)

**Search for:** `These maps determine the forecast.`

**Do:** REPLACE that paragraph (it is one long line in your file, ending `into $\hat{\mathbf{y}}_{\mathrm{FACTS}}$.`).

**New text:**
```latex
These maps determine the forecast. At each forecast origin, domain evidence is
assigned to the $K$ channels, and each agent returns an independent forecast.
A validation step scores each agent's historical forecasts against a
reference, the resulting skill scores set the fusion weights, and adaptive
fusion combines the agent forecasts into $\hat{\mathbf{y}}_{\mathrm{FACTS}}$.
Calibration is a separate, offline step that selects channel constants on
historical validation folds before the test period begins.
```

**Why:** calibration does not set the weights; the skill scores do.

## Step 13. Figure 1 caption

**Search for:** `fusion weights, which are recomputed when a new origin is reached. Calibration`

**Do:** REPLACE the full caption (four lines):
```latex
\caption{FACTS-MAS at one forecast origin. Each evidence channel returns an
independent horizon forecast. Skill relative to a naive reference sets the
fusion weights, which are recomputed when a new origin is reached. Calibration
acts on channel constants and may set a constant to zero.}
```

**New text:**
```latex
\caption{FACTS-MAS at one forecast origin. Each evidence channel returns an
independent horizon forecast. Skill relative to a naive reference, measured on
a validation window that ends at the training cutoff, sets the fusion weights.
Calibration acts offline on channel constants and may set a constant to zero.}
```

**Also:** if the image `facts-mas-architecture.png` draws calibration inside the per-origin loop or feeding the weights, redraw it so that calibration feeds constants into the channels from outside the loop.

## Step 14. Section 3.4: state the temperature

**Search for:** `the final allocation.` (the line just after the softmax equation, starting `where $\tau>0$`)

**Do:** INSERT AFTER `the final allocation.`

**New text:**
```latex
In the housing configuration $\tau=0.05$. This value was set by hand from the
magnitude of validation skill scores, not selected by a search; a
validation-selected temperature was tested and not adopted
(Section~\ref{sec:fusion_ablation}).
```

## Step 15. Section 3.4: last paragraph (when weights change)

**Search for:** `Weights are recomputed from rolling historical evidence rather than fixed`

**Do:** REPLACE that paragraph (four lines, ending `is permitted as new validation evidence becomes available.`).

**New text:**
```latex
Weights are re-estimated whenever the training window is extended rather than
fixed once for the entire evaluation period. In the backtest, each fold
estimates its weights from a 104-week validation window ending at the training
cutoff and holds them fixed over that fold's test window. Consequently,
$w_{k,t_1}^{(H)} \neq w_{k,t_2}^{(H)}$ is permitted across folds, and within
a fold when $t_1$ and $t_2$ fall in different regimes.
```

## Step 16. Section 3.5: the sufficiency rule actually used

**Search for:** `allows the level of adaptation to be constrained by historical support rather`

**Do:** INSERT AFTER the next line, which ends `always improves generalization.`

**New text:**
```latex
In the housing configuration, a regime-specific weight vector is used only
when the regime contains at least 13 distinct validation weeks; otherwise the
forecast falls back to the pooled weights. An earlier version of the
implementation had no such rule and assigned equal weights to every channel
when a regime had no validation data, which affected 780 test cells and
inflated the apparent benefit of regime conditioning.
```

## Step 17. Section 3.6: punctuation in the calibration equation

**Search for:** `\hat{\mathbf{y}}_f(\theta)`

**Do:** on the next line, `\right),` closes the equation. Change `\right),` to `\right).`

## Step 18. Section 3.6: disclose how constants were selected

**Search for:** `direct numerical adjustment is removed.`

**Do:** INSERT AFTER that line, as a new paragraph (leave one blank line first).

**New text:**
```latex
In the housing study, each constant was selected once using the validation
windows of all six folds, with a candidate promoted only if it improved
validation loss in at least five of the six folds. Because later folds'
validation windows postdate earlier folds' test windows, this selection is not
fully nested in time. The only adopted calibration result, a zero event-impact
scale, changes test accuracy by at most 0.002 percentage points.
```

## Step 19. Section 3.7: attribution equation

**Search for:** `For forecast origin $t$, the contribution of agent $k$ can be represented by`

**Do:** REPLACE from that line through the `\end{equation}` that follows (the block containing `C_{k,t}^{(H)} = w_{k,t}^{(H)} \hat{\mathbf{y}}_{k,t}^{(H)}`).

**New text:**
```latex
For forecast origin $t$, the contribution of agent $k$ to the forecasted
change from the last observed value is

\begin{equation}
C_{k,t}^{(H)}
=
w_{k,t}^{(H)}
\left(\hat{\mathbf{y}}_{k,t}^{(H)} - y_t\mathbf{1}\right),
\end{equation}
so that $\sum_k C_{k,t}^{(H)} = \hat{\mathbf{y}}_{\mathrm{FACTS},t}^{(H)} - y_t\mathbf{1}$.
A level-based share $w_{k,t}^{(H)}\hat{\mathbf{y}}_{k,t}^{(H)}$ would credit
an agent that merely repeats the last observed value with a share of the
forecast level even though it contributes nothing to the forecasted change;
the change-based form assigns such an agent a contribution of zero.
```

**Then keep** the next paragraph (`The set ... provides an interpretable summary ...`) unchanged.

## Step 20. Section 3.8 (Domain Adaptation)

No change.

---

# 4 CASE STUDIES

## Step 21. Section 4 opening paragraph

**Search for:** `Three cases---weekly U.S.\ housing inventory time series, energy demand`

**Do:** REPLACE the whole paragraph, from that line through the line ending `held-out testing rules do not.`

**New text:**
```latex
Two cases---weekly U.S.\ housing inventory and weekly unemployment insurance
claims---are analyzed by FACTS-MAS, and a third, energy demand, is specified
as a planned configuration only. They are used to answer the following
questions: First, does the fused forecast improve on simple and competitive
references? Second, which evidence channels add incremental skill, and does
that skill change with horizon or regime? Third, can the same decomposition
and fusion procedure be re-instantiated when the target and the raw evidence
change? The housing study answers the first two questions, and unemployment
claims are used for the third; the energy configuration is reported as a
channel map without results. All experiments follow one rolling-origin
protocol. Data and channel models may change by domain; the information
boundary, agent-level scoring, weight estimation, and held-out testing rules
do not.
```

## Step 22. Section 4.1: pre-test information sentence

**Search for:** `so fitting, calibration, lags, fusion weights, and the choice of weighting`

**Do:** REPLACE these two lines:
```latex
so fitting, calibration, lags, fusion weights, and the choice of weighting
rule use only pre-test information.
```

**New text:**
```latex
so model fitting, fusion weights, and the choice of weighting rule use only
pre-test information. Two exceptions in the development of the housing
configuration are disclosed and corrected. The seasonal smoothing window was
originally chosen on the test window of the last fold, including targets in
January and February 2026; it was re-selected on validation data alone,
nested by fold, and the reported results use the re-selected window. The
calibrated constants were selected across all folds' validation windows
(Section~\ref{sec:calibration}).
```

## Step 23. Section 4.1: after the embargo sentence

**Search for:** `weekly housing data it is 13 weeks.`

**Do:** INSERT AFTER that line.

**New text:**
```latex
Validation forecasts used to estimate fusion weights are restricted to
targets dated on or before the training cutoff, so that no validation target
falls inside the embargo. Test targets dated on or after January 1, 2026, the
start of the reserved period, are excluded from the backtest; that period is
used only for the holdout in Section~\ref{sec:holdout}.
```

## Step 24. Section 4.1: statistics paragraph after the MAPE/RMSE equation

**Search for:** `error \citep{hyndman2006accuracy},`

**Do:** find the `\end{equation}` a few lines below (it closes the MAPE/RMSE equation). INSERT AFTER that `\end{equation}`, as a new paragraph, before `Each agent is also scored alone`.

**New text:**
```latex
Accuracy differences are tested on the weekly cross-section of losses. For
each pair of methods, the panel Diebold--Mariano test of equal average
accuracy \citep{diebold1995comparing,qu2024comparing} uses a Newey--West
variance with lag $H-1$ \citep{newey1987simple}; per-MSA tests use the
small-sample correction of \citet{harvey1997testing} with the
Benjamini--Hochberg adjustment across the 15 MSAs
\citep{benjamini1995controlling}. Confidence intervals come from a moving
block bootstrap over whole weeks \citep{kunsch1989jackknife} with 5,000
replications and a block length of 13 weeks. Two methods are called
equivalent when the 90\% confidence interval of their MAPE difference lies
within $\pm 2\%$ of the reference method's MAPE. The model confidence set
\citep{hansen2011mcs} summarizes the comparison among all methods.
```

## Step 25. Section 4.1: define the ablations

**Search for:** `A positive value means that the channel contributes inside the combination.`

**Do:** INSERT AFTER that line.

**New text:**
```latex
Removing a channel renormalizes the remaining weights in each cell without
re-estimating them. Removing adaptive weighting replaces every cell's weights
with the fold's pooled weights, which removes both horizon and regime
conditioning, so this ablation alone cannot separate the two.
```

## Step 26. Section 4.1: weighting rules paragraph

**Search for:** `Four weighting rules are compared on the same forecasts: one weight vector`

**Do:** REPLACE from that line through the line ending `forecast origin and test error is also reported by horizon and regime.`

**New text:**
```latex
Two weighting rules are compared on the same forecasts: one pooled weight
vector per fold, and separate weights for each combination of regime and
horizon. For each fold and horizon, the rule with lower validation error is
selected, $g^{*}=\arg\min_{g}L_{\mathrm{val}}(g)$, and fixed before the test
interval. The regime is computed at the forecast origin, and test error is
also reported by horizon and regime. To check whether regime labels carry
information, the same pipeline is rerun with regime labels permuted in blocks
and with horizon-only weights that ignore the regime.
```

**Why:** the fourth rule ("adaptive granular") had been chosen against using test data, so it is removed from the paper.

## Step 27. Section 4.1: baselines paragraph

**Search for:** `Every compared method is scored at the same forecast origins and against the`

**Do:** REPLACE that whole paragraph, through the line ending `domain (Section~\ref{sec:cross_domain}).`

**New text:**
```latex
Every compared method is scored at the same forecast origins and against the
same realized targets. The references are the last observed value, a
seasonal-naive forecast, and the seasonally adjusted naive forecast of the M4
competition \citep{makridakis2020m4}. The statistical baselines are Holt's
linear-trend model, its damped form, factor-adjusted exponential smoothing
with and without damping, and a factor-adjusted Theta model
\citep{assimakopoulos2000theta}. The strongest single model chosen on
validation data is also reported, together with equal-weight combinations of
all channels and of the temporal and seasonal channels. The learned
baselines are the TimesFM~2.5 foundation model and a chain-of-thought
language-model forecast. The primary success check is that the fused
FACTS-MAS forecast improves on the last-observed-value reference over the
full test window; the stricter check is whether it improves on the strongest
single model. Design choices that used test data during development are
listed in the protocol paragraph above, and the final configuration,
hypotheses, and tests were fixed before the holdout was accessed.
Cross-domain transfer keeps the same architecture and learns a new evidence
mapping for each domain (Section~\ref{sec:cross_domain}).
```

**Why:** "All weights and constants are chosen without access to test outcomes" was not true.

---

# 4.2 HOUSING

## Step 28. Section 4.2 opening: say two channels were retired

**Search for:** `seasonality, metropolitan structure, and disaster declarations. The task is`

**Do:** INSERT the new text right after `disaster declarations.` on that line, before `The task is`.

**New text:**
```latex
Two of these channels, the intrinsic and event channels, calibrate to a null
quantitative effect and are retired from the final housing forecast; they are
reported as tested channels.
```

## Step 29. 4.2.1: data dates and the smoothed series

**Search for:** `Zillow's available weekly history begins on February 3, 2018 and extends`

**Do:** REPLACE these lines (the sentence before them, `The target variable is weekly for-sale housing inventory obtained from Zillow Research.`, stays):
```latex
Zillow's available weekly history begins on February 3, 2018 and extends
through June 20, 2026 in the dataset used for this study. The resulting target
panel contains 438 weekly observations per metropolitan statistical area (MSA)
and 6,570 MSA--week
observations overall.
```

**New text:**
```latex
Zillow reports this series as a four-week trailing average; the unsmoothed
weekly series is used as a robustness check. The panel begins on February 3,
2018. The backtest uses
observations through February 21, 2026. The holdout uses a fresh download
taken on October 20, 2026 with observations through October 10, 2026 (454
weekly observations per metropolitan statistical area (MSA) and 6,810
MSA--week observations); every value dated on or before February 21, 2026 in
that download equals the earlier vintage, so no revision affects the results.
```

## Step 30. 4.2.1: missing-value sentences

**Search for:** `audit log.` (the line `used during imputation, and the affected rows were retained in a separate audit log.`)

**Do:** INSERT AFTER `audit log.`

**New text:**
```latex
The unsmoothed series has 20 missing observations, filled the same way. No
missing observation falls in a test or holdout target.
```

## Step 31. 4.2.1: the "non-overlapping" passage

**Search for:** `The decomposition is deliberately non-overlapping where possible. In`

**Do:** REPLACE from that line through the line ending `agents from encoding the same seasonal mechanism by construction.`

**New text:**
```latex
The initial decomposition is deliberately non-overlapping: the 4-week
temporal agent does not model seasonality internally because seasonality is
represented by a separate evidence channel, which makes the incremental value
of each component easier to evaluate. At 8 and 13 weeks, however, the
temporal agent that survived validation is a factor-adjusted model that
reuses the seasonal channel's factors (see the temporal channel below), so
the two channels are not independent at those horizons.
```

## Step 32. Table 2 (channel table): rows

**Search for:** `\label{tab:housing_channels}`

**Do:** in that table, REPLACE every row between `\midrule` and `\bottomrule` (the five channel rows, from `Temporal` down to the `\\` after the Event row).

**New text:**
```latex
Temporal
&
Historical weekly inventory
&
4 weeks: ETS with additive trend and no seasonal component; 8 and 13 weeks:
damped-trend ETS on seasonally adjusted inventory (factor-adjusted)
\\

Exogenous
&
Mortgage rate, federal funds rate, CPI, unemployment
&
Lagged ridge-regression modifier with pre-specified lags
\\

Seasonal
&
Week-of-year inventory pattern
&
Ratio-to-moving-average seasonal factors (seasonally adjusted naive form)
\\

Intrinsic
&
Population, income, housing stock, density, hazard exposure
&
Scaled ridge regression with structural mean reversion; retired (calibrated
reversion cap is zero)
\\

Event
&
FEMA disaster declarations
&
LLM-assisted event interpretation with confidence gating; retired (calibrated
impact scale is zero)
\\
```

## Step 33. Temporal channel paragraph (replace all of it)

**Search for:** `\paragraph{Temporal channel}`

**Do:** REPLACE from `\paragraph{Temporal channel}` through the line ending `test-informed model overrides.`

**New text:**
```latex
\paragraph{Temporal channel}
The temporal channel models the endogenous trajectory of each MSA's inventory
series independently. At the 4-week horizon it uses exponential smoothing
with an additive trend and no seasonal component
\citep{holt2004exponential,winters1960forecasting,hyndman2002statespace,pan2011holtwinters},
$\hat{\mathbf{y}}_{\mathrm{AR}}=\mathrm{ETS}(y_{1:t};\text{trend}=\mathrm{add},\text{seasonal}=\varnothing)$.
At 8 and 13 weeks it uses a factor-adjusted damped-trend model: the series is
divided by the seasonal factors of the seasonal channel, a damped additive
trend \citep{gardner1985damped} is fitted to the adjusted series, and the
forecast is multiplied by the factors of the target weeks. This variant was
promoted by a rule fixed before the comparison was run: it had to improve
validation error and improve test error in at least five of the six folds;
it met the rule at 8 and 13 weeks but not at 4 weeks. Because this rule
consults test-fold accuracy, the holdout in Section~\ref{sec:holdout}
provides the clean test of the resulting configuration. Damping was chosen
over an undamped trend on validation data in every fold. Both models are
fitted separately for each MSA and forecast origin using only the available
historical target, and negative extrapolated values are clipped at zero.
A last-value persistence forecast, $\hat{y}^{\mathrm{naive}}_{t+h}=y_t$,
serves as the minimum baseline. Over all test origins, the undamped trend
model has MAPE 2.368\%, 5.430\%, and 10.225\% at 4, 8, and 13 weeks,
against 4.194\%, 7.225\%, and 10.814\% for the naive forecast, and it
improves on the naive forecast in 15, 14, and 11 of the 15 MSAs. Its
advantage therefore narrows sharply with horizon, and at 13 weeks it is
worse than the naive forecast in four of the six folds. The factor-adjusted
damped model has MAPE 1.974\%, 3.608\%, and 5.664\%.
```

**Why:** the old numbers (2.73 / 5.51 / 7.79 against 4.69 / 9.34 / 13.67) came from a superseded script that made one forecast per fold.

## Step 34. Exogenous channel: the lags sentence

**Search for:** `The initial lags were 6 weeks for the`

**Do:** REPLACE from `The initial lags were` through `5 weeks for unemployment.` (two lines).

**New text:**
```latex
The lags are pre-specified from the literature rather than fitted: 6 weeks
for the mortgage rate, 10 weeks for the federal funds rate, 10 weeks for CPI,
and 5 weeks for unemployment.
```

## Step 35. Exogenous channel: the p-value sentences

**Search for:** `specification contained 5,850 observations. Mortgage rate, the theoretically`

**Do:** REPLACE from `Mortgage rate, the theoretically` (middle of that line) through `5\% threshold ($p=0.042$).`

**New text:**
```latex
Mortgage rate, the theoretically most direct housing-finance variable,
was statistically indistinguishable from zero ($p=0.575$); unemployment was
also null ($p=0.548$), and CPI was borderline ($p=0.085$). The federal funds
rate had $p=0.042$, but this does not survive a Holm adjustment across the
four variables ($p=0.170$), a wild cluster bootstrap ($p=0.073$), or
Driscoll--Kraay standard errors that allow for the fact that all MSAs share
the same national regressor \citep{driscoll1998consistent} ($p=0.71$).
Used as a forecast on its own, the channel is no more accurate than the naive
forecast (MAPE 4.258\%, 7.297\%, and 10.919\%).
```

## Step 36. Seasonal channel: smoothing window

**Search for:** `The week-of-year profile is smoothed circularly so that the transition between`

**Do:** REPLACE from that line through `are then normalized to have mean one.` (three lines).

**New text:**
```latex
The week-of-year profile is smoothed with a seven-week circular moving
average, so that the transition between week 53 and week 1 is treated as
adjacent rather than as a boundary. Factors are then normalized to have mean
one. The window was selected on validation data, nested by fold, from
$\{1,3,5,7\}$; the value seven was chosen in every fold. An earlier window of
five had been chosen on the test window of the last fold and is not used.
Near the forecast origin the centered average is computed from the available
half-window only, so no factor uses data after the origin.
```

## Step 37. Seasonal channel: last sentence

**Search for:** `This formulation captures the annual shape of housing listings without`

**Do:** REPLACE that sentence (two lines, ending `the same mechanism.`).

**New text:**
```latex
This forecast is the seasonally adjusted naive method of the M4 competition
\citep{makridakis2020m4} with leakage-safe factors. On its own it has MAPE
2.890\%, 4.774\%, and 6.945\% at 4, 8, and 13 weeks.
```

## Step 38. Intrinsic channel: reversion cap and retirement

**Search for:** `with $\rho_{\max}=0.30$ in the current housing configuration. The quadratic`

**Do:** REPLACE from that line through `later in the forecast window.`

**New text:**
```latex
with $\rho_{\max}=0.30$ in the original housing configuration. The cap was
then calibrated on validation folds over
$\{0,0.10,0.20,0.30,0.45,0.60,0.80\}$, and zero was preferred in all six
folds (validation MAPE 7.923\% at zero against 10.259\% at 0.30). At
$\rho_{\max}=0$ the agent reproduces the naive forecast, so the channel is
retired from the housing forecast. The quadratic schedule is also not
consistent across horizons: the forecast for week $t+4$ closes 0.30 of the
gap when $H=4$ but 0.028 of it when $H=13$. Because the channel is retired,
this is documented rather than corrected. The static features use the 2024
Census vintage and 2016--2025 storm records for every fold; replacing the
channel's output with the naive forecast changes fused accuracy by at most
0.011 percentage points, so this vintage has no material effect.
```

## Step 39. Event channel: name the LLM

**Search for:** `The Event Agent is the only housing component that uses an LLM.`

**Do:** REPLACE that sentence.

**New text:**
```latex
The Event Agent is the only housing component that uses an LLM (gpt-4o-mini).
```

## Step 40. Event channel: duration sentences

**Search for:** `For the active-event duration, progressively shorter windows produced lower`

**Do:** REPLACE from that line through `showed that zero influence was best in all six validation folds.`

**New text:**
```latex
For the active-event duration, progressively shorter windows produced lower
validation error in all six folds. Once the impact scale is set to zero
(below), the duration no longer affects the forecast.
```

**Why:** the extended duration grid could not be reproduced, so it is dropped, as agreed.

## Step 41. Event calibration table (Table 3): new values

**Search for:** `\caption{Housing Event Agent impact-scale calibration.}`

**Do:** REPLACE the caption line and the `MAPE` row of that table.

**New caption:**
```latex
\caption{Housing Event Agent impact-scale calibration: mean validation MAPE.
Validation targets are restricted to dates on or before each fold's training
cutoff; an earlier version of this sweep allowed targets to extend up to 13
weeks past it, which changed the values by 1--2\% but not the ordering.}
```

**New MAPE row** (replaces `MAPE & \textbf{5.813} & 5.982 & ...`):
```latex
MAPE
& \textbf{5.704}
& 5.869
& 6.112
& 6.413
& 7.110
& 8.121
& 9.234
\\
```

## Step 42. Event channel: zero scale means the channel equals naive

**Search for:** `The scale of zero was preferred in 6 of 6 folds. Consequently, the production`

**Do:** INSERT AFTER `The scale of zero was preferred in 6 of 6 folds.`

**New text:**
```latex
With a zero scale, the agent's forecast equals the last observed value at
every origin, so the event channel reproduces the naive reference and is
retired from the final housing forecast.
```

## Step 43. Folds paragraph and Table 4: reserved window and holdout

**Search for:** `The housing experiment uses six expanding-window folds and reserves data from`

**Do:** REPLACE from that line through the line ending `window rather than at one forecast date per fold.`

**New text:**
```latex
The housing experiment uses six expanding-window folds and reserves data from
January 2026 onward. In an earlier version of the experiment, 8- and 13-week
forecasts issued near the end of the sixth test window had targets in January
and February 2026 (165 MSA--origin--horizon cells); these cells are excluded
from every backtest result reported here. The fold schedule is shown in
Table~\ref{tab:housing_folds}, together with the holdout described in
Section~\ref{sec:holdout}. Every fold contains at least a 13-week separation
between the final eligible training date and the test period, matching the
longest housing forecast horizon. Forecasts are evaluated at every eligible
weekly origin within each test window rather than at one forecast date per
fold.
```

**Also in Table 4:** after the fold 6 row (`6 & 2019-02-02 & 2025-02-01 & 2025-05-10 & 2025-11-22 \\`), INSERT:
```latex
\midrule
Holdout & 2019-02-02 & 2026-02-21 & 2026-02-28 & 2026-09-12 \\
```

**And change the caption** `\caption{Six-fold housing backtest schedule.}` to:
```latex
\caption{Housing backtest folds and holdout. Test start and end are the first
and last forecast origins.}
```

## Step 44. Regime paragraph: add the national-label caveat

**Search for:** `ensuring that it represents information available when the prediction is`

**Do:** INSERT AFTER the next line, which ends `made.`

**New text:**
```latex
Because the federal funds rate is national, all 15 MSAs share the same regime
at every date, and the sample contains one substantive hiking episode (June
2022 to September 2023) and three cutting episodes, so regime conditioning is
close to conditioning on a small number of historical periods.
```

## Step 45. The "Four fusion configurations" passage

**Search for:** `Four fusion configurations were investigated:`

**Do:** REPLACE from that line through the line ending `cells in which finer granularity helped and those in which it did not.` This includes the enumerate list and the paragraph after it.

**New text:**
```latex
Two fusion configurations are compared: pooled fold-level weighting, and
adaptive selection, for each horizon, between pooled weights and weights
conditioned on regime and horizon. The production configuration uses adaptive
selection with the 13-week data-sufficiency rule of
Section~\ref{sec:adaptive_context}. A finer, cell-level selection was also
explored during development, but that choice was informed by test-window
results and is not reported. Section~\ref{sec:fusion_ablation} tests whether
the regime labels carry information.
```

---

# 4.2.2 FORECASTING RESULTS

## Step 46. Primary results text and Table 5 (replace)

**Search for:** `Table~\ref{tab:housing_primary} reports the primary performance comparison`

**Do:** REPLACE from that line through the `\end{table}` that closes Table 5 (`tab:housing_primary`).

**New text:**
```latex
Table~\ref{tab:housing_primary} reports test accuracy over the six backtest
folds for FACTS-MAS, its references, and the statistical baselines, on both
Zillow series. FACTS-MAS outperforms the persistence reference at all three
horizons: the panel Diebold--Mariano statistic is $-6.9$, $-6.3$, and $-5.7$
at 4, 8, and 13 weeks ($p<10^{-4}$), the difference is significant in all 15
MSAs at every horizon, and the 95\% confidence intervals of the MAPE
difference are $[-2.66,-1.42]$, $[-4.72,-2.52]$, and $[-7.11,-3.53]$. The
relative reduction is 49.5\%, 49.9\%, and 47.0\% on the smoothed series and
23.9\%, 34.5\%, and 37.1\% on the unsmoothed series. The strongest single
model, factor-adjusted damped exponential smoothing (FA-ETS-d), is at least as
accurate as FACTS-MAS at every horizon. FACTS-MAS is statistically equivalent
to it at 8 weeks (90\% interval of the difference $[-0.001,0.030]$, within the
$\pm0.072$ margin), less accurate at 4 weeks by 0.144 (90\% interval
$[0.022,0.261]$; Diebold--Mariano $p=0.12$), and marginally less accurate at
13 weeks by 0.067 (95\% interval $[0.001,0.152]$; $p=0.09$). At 8 weeks the
90\% model confidence set contains only FA-ETS-d, FACTS-MAS, and the
five-channel variant of FACTS-MAS; at 13 weeks it excludes only the naive
forecast. The fused forecast therefore improves strongly on the references
but does not improve on the best single model.

\begin{table}[t]
\centering
\caption{Backtest MAPE (\%) over six folds, all test origins, targets before
January 1, 2026. FA-ETS-d: factor-adjusted damped ETS. Lower is better; bold
marks the lowest value in each column.}
\label{tab:housing_primary}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Smoothed series}}
& \multicolumn{3}{c}{\textbf{Unsmoothed series}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Method} & 4 wk & 8 wk & 13 wk & 4 wk & 8 wk & 13 wk \\
\midrule
Naive (last value)        & 4.194 & 7.225 & 10.814 & 4.785 & 7.882 & 11.438 \\
Seasonal naive            & 20.771 & 20.513 & 20.597 & -- & -- & -- \\
Seasonally adjusted naive & 2.890 & 4.774 & 6.945 & 3.626 & 5.408 & 7.489 \\
Holt (temporal channel)   & 2.368 & 5.430 & 10.225 & 4.317 & 7.901 & 13.022 \\
Theta (factor-adjusted)   & 2.933 & 4.842 & 7.047 & -- & -- & -- \\
FA-ETS-d                  & \textbf{1.974} & \textbf{3.608} & \textbf{5.664} & \textbf{3.402} & \textbf{5.105} & \textbf{7.164} \\
FACTS-MAS                 & 2.118 & 3.622 & 5.731 & 3.642 & 5.159 & 7.189 \\
\bottomrule
\end{tabular}
\end{table}
```

**Why:**
- The old Table 5 (2.243 / 5.038 / 7.020) was the July system and included the reserved-window cells.
- Seasonal naive and Theta were not run on the unsmoothed series, hence the dashes.

## Step 47. Fold-1 TimesFM/CoT comparison and Table 6 (replace)

**Search for:** `A separate matched comparison was performed on the complete Fold-1 test`

**Do:** REPLACE from that line through the `\end{table}` that closes Table 6 (`tab:housing_strong_baselines`).

**New text:**
```latex
Table~\ref{tab:housing_strong_baselines} compares FACTS-MAS with Google's
TimesFM~2.5 time-series foundation model \citep{das2024timesfm}, run
zero-shot with a 52-week context at every test origin of all six folds.
TimesFM improves on the naive reference at every horizon ($p<10^{-4}$), but
FACTS-MAS is more accurate than TimesFM at every horizon, by 0.243, 1.239,
and 2.539 percentage points (Diebold--Mariano $p=0.006$, $0.0001$, and
$0.0004$; significant in 7, 10, and 10 of the 15 MSAs). FA-ETS-d is likewise
more accurate than TimesFM at every horizon. The chain-of-thought
language-model forecast is evaluated on the holdout only
(Section~\ref{sec:holdout}), because the backtest period falls inside the
training data of the language models available for the comparison.

\begin{table}[t]
\centering
\caption{Backtest MAPE (\%) over six folds against the TimesFM~2.5
foundation model, smoothed series.}
\label{tab:housing_strong_baselines}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & 4 weeks & 8 weeks & 13 weeks \\
\midrule
Naive       & 4.194 & 7.225 & 10.814 \\
TimesFM 2.5 & 2.361 & 4.860 & 8.269 \\
FA-ETS-d    & \textbf{1.974} & \textbf{3.608} & \textbf{5.664} \\
FACTS-MAS   & 2.118 & 3.622 & 5.731 \\
\bottomrule
\end{tabular}
\end{table}
```

**Why:**
- The old Table 6 used fold 1 only and the July system.
- Its CoT forecast came from a model whose training data covers 2021.

## Step 48. Regime paragraphs (replace) and the new holdout subsection

**Search for:** `The initial pooled fusion strategy performed well overall but exposed a`

**Do:** REPLACE from that line through the line ending `aggregate metrics can hide context-specific failures.` This includes both arrays.

**New text:**
```latex
Errors stratified by monetary-policy regime show no regime in which the final
forecast fails. At the 13-week horizon,
\[
\begin{array}{c|ccc}
\text{Regime}
&
\text{Naive}
&
\text{FA-ETS-d}
&
\text{FACTS-MAS}
\\
\hline
\text{hiking}  & 8.059  & 5.530 & 5.552 \\
\text{cutting} & 13.043 & 6.091 & 6.090 \\
\text{stable}  & 11.628 & 5.666 & 5.759
\end{array}
\]
so FACTS-MAS improves on the naive reference in every regime, and FA-ETS-d,
which uses no regime information, is equally accurate in every regime. An
earlier version of the system appeared to lose to the naive reference in the
hiking and cutting regimes and to be repaired by regime-aware weighting; that
pattern came from an older seasonal channel and, in the cutting regime, from
the equal-weight fallback described in Section~\ref{sec:adaptive_context}, and
it does not appear in the current configuration.

\subsubsection{Holdout Evaluation}
\label{sec:holdout}

After the configuration was frozen, it was evaluated once on data that had not
been accessed during development: forecast origins from February 28 to
September 12, 2026 (29, 25, and 20 weekly origins at 4, 8, and 13 weeks for
each of the 15 MSAs). The fusion weights were estimated from the 104-week
validation window ending on February 21, 2026. Four hypotheses, the tests, and
the reporting rules were committed before the holdout data were downloaded:
FACTS-MAS improves on the naive reference at every horizon (H1); FA-ETS-d does
too (H2); FACTS-MAS and FA-ETS-d are equivalent at 8 and 13 weeks (H3); and
FA-ETS-d is at least as accurate as FACTS-MAS at 4 weeks (H4). With 20 origins
the 13-week results are reported as descriptive only. The bootstrap block
length equals the horizon, and at 13 weeks it is capped at half the number of
weeks. The chain-of-thought baseline uses the NEXUS prompt with gpt-5.6-luna
(reasoning effort low), whose documented knowledge cutoff, February 16, 2026,
precedes the first holdout target; the model's own answer to a cutoff question
was June 2024. The prompt itself states a January 2025 cutoff, a wording
inherited from the NEXUS template.

Table~\ref{tab:holdout} reports the holdout accuracy and
Table~\ref{tab:holdout_tests} the hypothesis tests. All four hypotheses are
supported. FACTS-MAS reduces MAPE relative to the naive forecast by 60\%,
65\%, and 64\% on the smoothed series and by 39\%, 50\%, and 54\% on the
unsmoothed series. It is equivalent to FA-ETS-d at 8 weeks and less accurate
than FA-ETS-d at 4 weeks by 0.105 percentage points. The chain-of-thought
forecast is competitive: its MAPE is higher than that of FA-ETS-d and of
FACTS-MAS at every horizon, but neither difference is statistically
significant (against FA-ETS-d, $p=0.38$, $0.43$, and $0.60$; against
FACTS-MAS, in a comparison added after the results were seen, $p=0.90$,
$0.43$, and $0.54$). TimesFM is
significantly less accurate than FA-ETS-d at 4 and 8 weeks ($p=0.016$ and
$p<10^{-4}$). On the unsmoothed series FACTS-MAS at 4 weeks coincides with the
seasonally adjusted naive forecast, and at 8 weeks it is slightly more
accurate than FA-ETS-d ($-0.038$, $p=0.031$); because the seasonally adjusted
naive forecast is equally accurate there, this does not indicate a benefit of
fusion over a single model. With a two-week publication lag, every method's
error rises (FACTS-MAS 2.270\%, 2.944\%, and 4.000\%; naive 5.474\%, 7.953\%,
and 11.110\%), and no conclusion changes. In the lag-two run, 27 of 1,110
chain-of-thought forecasts (2.4\%) were lost to a network interruption; the
tests for that run were computed with and without the chain-of-thought
forecast. The holdout period was easier for every method than the backtest,
so absolute errors should be compared within a period, not across periods.

\begin{table}[t]
\centering
\caption{Holdout MAPE (\%), forecast origins February 28 to September 12,
2026, no publication lag. 13-week values are descriptive. CoT: chain-of-thought
forecast (gpt-5.6-luna), run on the smoothed series only.}
\label{tab:holdout}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Smoothed series}}
& \multicolumn{3}{c}{\textbf{Unsmoothed series}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Method} & 4 wk & 8 wk & 13 wk & 4 wk & 8 wk & 13 wk \\
\midrule
Naive (last value)        & 3.122 & 5.445 & 8.438 & 3.355 & 5.306 & 7.925 \\
Seasonal naive            & 8.534 & 8.534 & 8.550 & 8.593 & 8.601 & 8.622 \\
Seasonally adjusted naive & 1.492 & 2.377 & 3.393 & 2.053 & 2.677 & \textbf{3.564} \\
TimesFM 2.5               & 1.410 & 2.602 & 4.457 & 2.464 & 3.689 & 5.355 \\
CoT (gpt-5.6-luna)        & 1.259 & 2.225 & 3.268 & -- & -- & -- \\
FA-ETS-d                  & \textbf{1.138} & 1.936 & 3.080 & \textbf{2.028} & 2.709 & 3.736 \\
FACTS-MAS                 & 1.243 & \textbf{1.929} & \textbf{3.045} & 2.053 & \textbf{2.671} & 3.644 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Pre-registered holdout hypotheses, smoothed series, no publication
lag. Differences are in MAPE percentage points with 95\% block-bootstrap
intervals (90\% intervals for the equivalence test, H3). 13-week rows are
descriptive.}
\label{tab:holdout_tests}
\begin{tabular}{llccl}
\toprule
\textbf{Hypothesis} & \textbf{Horizon} & \textbf{Difference} & \textbf{Interval} & \textbf{Result} \\
\midrule
H1: FACTS-MAS $<$ naive  & 4 wk  & $-1.879$ & $[-3.031,-0.897]$ & supported \\
                          & 8 wk  & $-3.516$ & $[-5.681,-1.705]$ & supported \\
                          & 13 wk & $-5.393$ & $[-7.681,-3.071]$ & supported \\
H2: FA-ETS-d $<$ naive   & 4 wk  & $-1.984$ & $[-3.081,-1.031]$ & supported \\
                          & 8 wk  & $-3.509$ & $[-5.693,-1.680]$ & supported \\
                          & 13 wk & $-5.358$ & $[-7.642,-3.028]$ & supported \\
H3: equivalent to FA-ETS-d & 8 wk & $-0.007$ & $[-0.029,0.014]$ & equivalent ($\pm0.039$) \\
                          & 13 wk & $-0.035$ & $[-0.048,-0.035]$ & equivalent ($\pm0.062$) \\
H4: FA-ETS-d $\le$ FACTS-MAS & 4 wk & $+0.105$ & $[-0.007,0.182]$ & supported \\
\bottomrule
\end{tabular}
\end{table}
```

**Note:** the new section uses `\cmidrule`, which comes with `booktabs`; you already load it.

## Step 49. Limits paragraphs

**49a. Search for:** `every historical date. The hazard index is a partial structural proxy: NOAA`

**Do:** INSERT AFTER `every historical date.`

```latex
The target is Zillow's smoothed series, a four-week trailing average, which
makes every method look more accurate than on the unsmoothed series; both are
reported.
```

**49b. Search for:** `zeroed its quantitative effect, but the gap remains. The Intrinsic Agent's`

**Do:** REPLACE from `The Intrinsic Agent's` (middle of that line) through `held-out test performance.` (end of the paragraph).

**New text:**
```latex
Two design choices made during development used test-window evidence (the
original seasonal smoothing window and a cell-level fusion rule); both were
replaced or removed, and the holdout was evaluated only after the
configuration was frozen.
```

---

# 4.3 and 4.4 ENERGY AND UNEMPLOYMENT

## Step 50. Energy: state that no results are reported

**Search for:** `The intended channel map for this domain is` (the first occurrence, under Energy Forecasting)

**Do:** INSERT AFTER the next line, which ends `Table~\ref{tab:energy_mapping}.`

**New text:**
```latex
No energy experiments are reported in this paper; the mapping below records
the planned configuration.
```

## Step 51. Unemployment: replace the "planned" text with results

**51a. Search for:** `\caption{Planned FACTS-MAS evidence-channel mapping for unemployment`

**Do:** REPLACE the caption (two lines, ending `forecasting.}`).

```latex
\caption{FACTS-MAS evidence-channel mapping for unemployment insurance claims.
The temporal, exogenous, and seasonal channels are implemented; the intrinsic
and event channels are not.}
```

**51b. Search for:** `$\mathcal{C}^{(\mathrm{unemployment})}=(\mathcal{E}^{(\mathrm{unemployment})}$,`

**Do:** REPLACE from the line before it (`The configuration is`) through the end of that paragraph (`downweight empty channels.`).

**New text:**
```latex
The target is weekly initial unemployment insurance claims, not seasonally
adjusted, for 15 U.S.\ states from January 2013, obtained from FRED. The
temporal and seasonal channels use the housing implementations unchanged,
including the seven-week seasonal window, and the exogenous channel uses the
federal funds rate and CPI with the housing lags. The housing recipe is
applied without re-tuning: the same six folds, embargo, validation window,
data-sufficiency rule, temperature, and promotion outcome. The window of
pandemic-era claims, from March 14, 2020 to September 4, 2021 (the start of
the national surge to the expiry of the federal pandemic programs), is
excluded from skill estimation and scoring but kept for model fitting; this
window was fixed from policy dates before any result was computed. The
protocol was written before any unemployment result was computed.

Table~\ref{tab:ui_results} reports the results. The fused forecast improves
on the naive reference at every horizon ($p=0.002$, $0.0004$, and $0.0004$)
and on damped FA-ETS at 8 and 13 weeks ($p=0.0009$ and $0.005$), but validation
selected the undamped trend in every fold, and the undamped factor-adjusted
model alone is slightly more accurate than the fused forecast at every horizon.
In unemployment claims, the frozen recipe beats naive and the damped FA-ETS
reference, but not undamped FA-ETS, which is the better reference for that
series. In neither domain is there evidence that fusion beats the best single
model. The channel weights differ from housing, as the transfer hypothesis
expects: with the temporal channel that ignores seasonality, the seasonal
channel receives about three quarters of the weight at every horizon, and the
exchange of weight between the temporal and seasonal channels seen across
horizons in housing does not appear. Including the pandemic window raises
every method's MAPE by 0.7 to 2.2 points. Because an earlier pilot evaluation
scored claims data through August 2026, a clean holdout for this domain is not
available.

\begin{table}[t]
\centering
\caption{Unemployment insurance claims: test MAPE (\%) over six folds, pandemic
window excluded. The undamped FA-ETS row was added after the other results
were seen and is descriptive.}
\label{tab:ui_results}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & 4 weeks & 8 weeks & 13 weeks \\
\midrule
Naive                        & 18.816 & 21.963 & 24.716 \\
Seasonally adjusted naive    & 17.836 & 19.294 & 20.613 \\
FA-ETS-d (damped)            & 17.868 & 19.411 & 20.791 \\
FA-ETS (undamped)            & \textbf{17.71} & \textbf{19.18} & \textbf{20.55} \\
FACTS-MAS (frozen recipe)    & 17.863 & 19.233 & 20.578 \\
\bottomrule
\end{tabular}
\end{table}
```

[Sources: phase6_20261016.md F6.5-F6.7; ui_protocol.md; claims C21. The undamped row has two decimals because that is how the addendum reported it.]

---

# 5 DISCUSSION

## Step 52. 5.1.1: summary of channel results

**Search for:** `experiments show that pattern clearly. The temporal channel improves`

**Do:** REPLACE from `The temporal channel improves` (middle of that line) through `harmful before calibration.` (two lines down).

**New text:**
```latex
The temporal channel improves on persistence at 4 weeks but its advantage
narrows with horizon unless seasonality is removed first; seasonality helps at
every horizon; the tested macroeconomic magnitude channel adds little robust
incremental value; the intrinsic reversion calibrates to zero; and the initial
event-severity magnitude mapping is harmful before calibration.
```

## Step 53. 5.1.2: macro panel table and sentence

**53a. Search for:** `\caption{Final housing Macro agent signal validation.}`

**Do:** REPLACE the whole table, from the `\begin{table}[t]` just above that caption through its `\end{table}`.

```latex
\begin{table}[t]
\centering
\caption{Final housing Macro agent signal validation: $p$-values for the
original panel (MSA and year--quarter fixed effects, MSA-clustered errors),
with a Holm adjustment across the four variables, a wild cluster bootstrap
(Webb weights, 9,999 draws), and Driscoll--Kraay errors with MSA fixed effects.
$n=5{,}850$.}
\label{tab:macro_panel}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Variable} & \textbf{Lag} & \textbf{Coef.} & \textbf{Clustered} & \textbf{Holm} & \textbf{Wild bootstrap} & \textbf{Driscoll--Kraay} \\
\midrule
Mortgage rate & 6 weeks  &  0.0075 & 0.575 & 1.000 & 0.611 & 0.973 \\
Federal funds & 10 weeks & -0.0129 & 0.042 & 0.170 & 0.073 & 0.712 \\
CPI           & 10 weeks & -0.0047 & 0.085 & 0.254 & 0.086 & 0.463 \\
Unemployment  & 5 weeks  & -0.0015 & 0.548 & 1.000 & 0.577 & 0.264 \\
\bottomrule
\end{tabular}
\end{table}
```

**53b. Search for:** `Only federal funds crosses the conventional 5\% threshold; mortgage rates,`

**Do:** REPLACE that sentence (two lines, ending `are effectively null.`).

```latex
Only federal funds crosses the conventional 5\% threshold under clustered
errors, and it does not survive the multiple-testing adjustment, the wild
bootstrap, or errors that allow for the common national regressor; mortgage
rates, the most direct theoretical housing-finance channel, are null under
every specification.
```

## Step 54. 5.1.3: duplicate event table

**Search for:** `(Table~\ref{tab:event_scale_analysis}).`

**Do:** change `tab:event_scale_analysis` to `tab:event_calibration` in that reference.

## Step 55. 5.1.3: delete the duplicate table and fix the duration sentence

**55a. Search for:** `\caption{Event impact-scale calibration across housing validation folds.}`

**Do:** DELETE that entire table, from the `\begin{table}[t]` just above it through its `\end{table}`. It duplicates Table 3, which now has the corrected values.

**55b. Search for:** `MAPE, and $\lambda=0$ is preferred in all six folds. Calibrating the`

**Do:** REPLACE from `Calibrating the` (end of that line) through `zero quantitative event influence again performed best.`

```latex
With the scale at zero, the declaration-active window no longer affects the
forecast, and the event channel's output equals the naive reference.
```

## Step 56. 5.1.5: Adaptive fusion and its limits (rewrite)

**Search for:** `\paragraph{Regime-aware attribution.}`

**Do:** REPLACE from `\paragraph{Regime-aware attribution.}` through the line ending `the data-sufficiency fallback remains a future extension.` This removes:
- the Regime-aware attribution paragraph;
- the Comparison of fusion strategies paragraph, with its list and Table 11;
- the When more adaptation hurts paragraph, with Table 12;
- the old Data sufficiency paragraph.

**New text:**
```latex
\paragraph{Does regime conditioning carry information?}
\label{sec:regime_attribution}

Adaptive weighting chooses, for each fold and horizon, between pooled weights
and weights conditioned on regime and horizon. Its apparent 13-week benefit
was tested in three ways on the five-channel configuration, before the
temporal channel was replaced. First, with regime labels permuted in blocks,
46\% of 200 permutations at 8 weeks and 27\% at 13 weeks were at least as
accurate as the true labels. Second, weights conditioned on horizon alone,
with no regime, matched the regime-conditioned weights at 8 weeks (4.468\%
against 4.468\%) and were more accurate at 13 weeks (6.826\% against
6.987\%). Third, 60--75\% of the 13-week benefit came from a single fold, in
the stable regime. We therefore find no evidence that the regime labels carry
predictive information beyond horizon conditioning; with a national regime
variable and one substantive hiking episode in the sample, the regime
partition is close to a partition by historical period.

\paragraph{Data sufficiency.}
\label{sec:data_sufficiency}

Context-aware adaptation should depend on both estimated performance and
evidence volume. Let $n_{H,r}^{(f)}$ be the number of distinct validation
weeks for horizon $H$, regime $r$, and fold $f$. The production rule uses
regime-conditioned weights only if $n_{H,r}^{(f)}\geq 13$ and otherwise falls
back to the pooled weights. An earlier implementation had no such rule: when a
regime was absent from a validation window, every channel, including the
channels that reproduce the naive forecast, received a weight of 0.2. That
fallback affected 780 test cells, accounted for about one fifth of the
apparent 13-week benefit of regime weighting, and produced the apparent repair
of the cutting regime reported in an earlier version of this work.

\paragraph{Fusion versus the strongest single model.}
\label{sec:fusion_vs_single}

Once the temporal channel is seasonally adjusted, validation gives it 0.97 of
the weight at 8 weeks and 0.92 at 13 weeks, the seasonal channel receives most
of the remainder, and the exogenous channel receives almost none. The fused
forecast is then statistically equivalent to the factor-adjusted model alone
at 8 weeks and slightly less accurate at 4 and 13 weeks, in the backtest and on
the holdout. Three further candidates, each registered before it was run, did
not change this: a listing-flow channel built from Zillow new listings, newly
pending sales, and price-cut shares; fusion of deviations from the naive
forecast; and a validation-selected temperature with shrinkage toward equal
weights. None met the promotion rule, and the best was 0.002 percentage points
behind FA-ETS-d at 4 weeks ($p=0.42$). In the housing domain, the value of the
fusion step is therefore that it selects and recomposes the useful channels,
not that it combines them into a forecast more accurate than the best of them.
```

**Note:** keep the heading `\subsubsection{Adaptive Fusion and Its Limits}` and its `\label{sec:fusion_ablation}` line, which sit just above the replaced range. Other steps refer to that label.

## Step 57. 5.2: full-window evaluation paragraph

**Search for:** `The baseline experiments also exposed a significant evaluation artifact.`

**Do:** REPLACE from that line through the line ending `test interval.` (the end of that paragraph).

**New text:**
```latex
The baseline experiments also exposed a significant evaluation artifact.
An earlier diagnostic evaluated one forecast origin per fold and horizon, and
its comparison of the temporal channel with the naive forecast scored
horizons that fell inside the embargo. Under that design the 13-week forecast
appeared to lose to naive in most MSAs. A single forecast date is a poor
estimator of general forecasting performance, so every result reported here
uses all eligible forecast origins in each test window: 2,610 MSA--origin
pairs per horizon in the backtest and 29 weekly origins per MSA in the holdout.
```

## Step 58. 5.3 and 5.4: what the results imply, and the transfer hypotheses

**58a. Search for:** `Adaptive fusion improves on persistence at every evaluated horizon, and the`

**Do:** REPLACE from that line through `FACTS-MAS treats a null channel as an admissible outcome.` (the end of that paragraph).

```latex
FACTS-MAS improves on persistence at every evaluated horizon, on both Zillow
series, in the backtest and on the holdout, and it is more accurate than the
TimesFM foundation model. On the holdout a chain-of-thought language-model
forecast comes close to it, with no statistically significant difference.
It is not more accurate than the strongest single
model, factor-adjusted damped exponential smoothing, which it matches at 8
weeks. The more important architectural point is that the final forecast is
the product of tests that each channel had to pass. Temporal dynamics and
seasonality carry skill; the tested macro magnitude specification is largely
null; the intrinsic reversion and the event-magnitude map calibrate to zero.
That pattern is intentional. Available evidence is not the same as validated
predictive evidence: the first determines which agents can be proposed, and
the second determines how strongly they may move the forecast. A less
disciplined process could keep searching lags and prompts until a favorable
statistic appears; FACTS-MAS treats a null channel as an admissible outcome.
```

**58b. Search for:** `Regime analysis shows why aggregate MAPE is not enough. Pooled fold weights`

**Do:** REPLACE from `Regime analysis shows why aggregate MAPE is not enough.` through the end of that paragraph (`uncommon in the validation sample.`).

```latex
Regime analysis shows why apparent context effects need a placebo. An earlier
version of the system seemed to fail in the hiking and cutting regimes and to
be repaired by regime-aware weighting, but the failure disappeared once the
seasonal channel was estimated properly, the repair was partly an artifact of
an equal-weight fallback, and randomly permuted regime labels did as well as
the true ones. Context-specific weighting reduces the observations behind
each decision, and the rare regimes where adaptation looks most attractive
often have the least support. The tradeoff is context specificity versus
estimation reliability. Energy or unemployment applications will meet the same
constraint whenever extreme weather or recessions are uncommon in the
validation sample.
```

**58c. Transfer hypotheses. Search for:** `After that reconfiguration, adaptive fusion should remain competitive with`

**Do:** REPLACE from that line through `Sections~\ref{sec:energy} and~\ref{sec:unemployment}.`

```latex
After that reconfiguration, adaptive fusion should remain competitive with
the new domain's simple reference and strong standalone models.

The unemployment-claims study (Section~\ref{sec:unemployment}) supports H1:
the claims series was written in the same channels and contracts, and the
housing code ran unchanged. It supports H2: the learned weights differ from
housing, with the seasonal channel carrying most of the weight. It supports
H3 only in part: the fused forecast beats the naive reference and the damped
factor-adjusted model, but not the undamped factor-adjusted model, which
validation selected and which is the better single model for that series. The
energy configuration in Section~\ref{sec:energy} is planned and untested.
```

---

# 6 CONCLUSION

## Step 59. Conclusion, paragraph 2

**Search for:** `On weekly U.S.\ housing inventory across 15 metropolitan areas, the`

**Do:** REPLACE from that line through the end of the paragraph (`adaptive fusion procedure remain the same.`).

**New text:**
```latex
On weekly U.S.\ housing inventory across 15 metropolitan areas, the frozen
configuration attains backtest MAPE of 2.118\%, 3.622\%, and 5.731\% at 4,
8, and 13 weeks on Zillow's smoothed series, against 4.194\%, 7.225\%, and
10.814\% for last-value persistence, and 1.243\%, 1.929\%, and 3.045\% on a
pre-registered holdout, against 3.122\%, 5.445\%, and 8.438\%. The gains are
smaller but still significant on the unsmoothed series. FACTS-MAS is more
accurate than the TimesFM~2.5 foundation model; on the holdout its error is
lower than that of a chain-of-thought language-model forecast built on the
NEXUS prompt, but not significantly so. It is not more accurate than a
factor-adjusted damped exponential smoothing model, with which it is
equivalent at 8 weeks. Those results coexist with null and rejected channels:
the tested macro magnitude specification adds little short-horizon skill,
validation removes the intrinsic reversion and the cardinal event-severity
adjustment, and regime-aware weighting shows no benefit beyond horizon
conditioning once a placebo is applied. The contribution of FACTS-MAS is
therefore the protocol itself: every channel is tested against a reference,
calibration may return a null, the final forecast is recomposed from the
channels that survive, and the result is checked on data held out until the
configuration is frozen. Transferring the method to a different domain
requires a new evidence map, new channel models where needed, and new
validation-based calibration and weights, but the FACTS-MAS architecture, the
shared interface, leakage-safe scoring, and adaptive fusion procedure remain
the same.
```

---

# BIBLIOGRAPHY AND FINAL CHECK

## Step 60a. Add these entries to `references.bib`

```bibtex
@article{gardner1985damped,
  author  = {Gardner, Everette S. and McKenzie, Ed},
  title   = {Forecasting Trends in Time Series},
  journal = {Management Science},
  volume  = {31}, number = {10}, pages = {1237--1246}, year = {1985}
}
@article{assimakopoulos2000theta,
  author  = {Assimakopoulos, V. and Nikolopoulos, K.},
  title   = {The Theta Model: A Decomposition Approach to Forecasting},
  journal = {International Journal of Forecasting},
  volume  = {16}, number = {4}, pages = {521--530}, year = {2000}
}
@article{diebold1995comparing,
  author  = {Diebold, Francis X. and Mariano, Roberto S.},
  title   = {Comparing Predictive Accuracy},
  journal = {Journal of Business \& Economic Statistics},
  volume  = {13}, number = {3}, pages = {253--263}, year = {1995}
}
@article{harvey1997testing,
  author  = {Harvey, David and Leybourne, Stephen and Newbold, Paul},
  title   = {Testing the Equality of Prediction Mean Squared Errors},
  journal = {International Journal of Forecasting},
  volume  = {13}, number = {2}, pages = {281--291}, year = {1997}
}
@article{newey1987simple,
  author  = {Newey, Whitney K. and West, Kenneth D.},
  title   = {A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix},
  journal = {Econometrica},
  volume  = {55}, number = {3}, pages = {703--708}, year = {1987}
}
@article{benjamini1995controlling,
  author  = {Benjamini, Yoav and Hochberg, Yosef},
  title   = {Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {57}, number = {1}, pages = {289--300}, year = {1995}
}
@article{kunsch1989jackknife,
  author  = {K{\"u}nsch, Hans R.},
  title   = {The Jackknife and the Bootstrap for General Stationary Observations},
  journal = {The Annals of Statistics},
  volume  = {17}, number = {3}, pages = {1217--1241}, year = {1989}
}
@article{hansen2011mcs,
  author  = {Hansen, Peter R. and Lunde, Asger and Nason, James M.},
  title   = {The Model Confidence Set},
  journal = {Econometrica},
  volume  = {79}, number = {2}, pages = {453--497}, year = {2011}
}
@article{driscoll1998consistent,
  author  = {Driscoll, John C. and Kraay, Aart C.},
  title   = {Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data},
  journal = {Review of Economics and Statistics},
  volume  = {80}, number = {4}, pages = {549--560}, year = {1998}
}
@article{qu2024comparing,
  author  = {Qu, Ritong and Timmermann, Allan and Zhu, Yinchu},
  title   = {Comparing Forecasting Performance with Panel Data},
  journal = {International Journal of Forecasting},
  volume  = {40}, number = {3}, pages = {918--941}, year = {2024}
}
```

**Check:** I am sure of the first nine entries. Check the volume and pages of `qu2024comparing` on the journal's page before posting.

## Step 60b. Leftover check (search each; every one should now return nothing)

| Search for | Should be gone because |
|---|---|
| `2.243` | old headline |
| `5.038` | old headline |
| `7.020` | old headline |
| `11.168` | old naive (included reserved cells) |
| `7.371` | old naive |
| `9.155` | old fold-1 number |
| `5.813` | old event table |
| `7.645` | old granular number |
| `crosses the conventional` | must appear only in Step 53b's new sentence |
| `corrected these failures` | regime claim withdrawn |
| `stable across all six folds` | not true |
| `adaptive granular` / `Adaptive granular` | removed |
| `NEXUS-CoT` / `NEXUS-style CoT` | replaced by the holdout CoT |
| `14 of 15 MSAs at 4 and 8 weeks` | old ETS diagnostic |
| `should eventually` | rho calibration was done |
| `without access to test outcomes` | not true |
| `recomputed when a new origin` | weights are per fold |

Then compile. Check that there are no undefined references (`??`) and no duplicate labels (see the note in Step 56).
