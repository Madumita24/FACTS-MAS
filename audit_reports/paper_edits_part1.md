# Paper edits, part 1 (from the start of the paper to the Event channel paragraph)

For: Umang, applying edits in Overleaf. The source was truncated at "In practice, the LLM largely behaved as an or". Part 2 will cover the rest (Event channel to end: results tables, discussion, unemployment, conclusion, appendix).

How to read this file:
- Each edit gives the **location**, the **current text** (first words, enough to find it with Ctrl+F), the **action**, and the **new LaTeX**.
- Replacements follow the paper's own style: formal, third person for the method, "FACTS-MAS" as the subject, no bullet lists in running text, and numbers with three decimals and a percent sign in the text.
- Numbers are from `audit_reports/` (sources are given in brackets; do not paste the brackets).

Backtest numbers use the test cells with targets dated before January 1, 2026: 2,610 / 2,565 / 2,490 cells at 4 / 8 / 13 weeks.

## Decisions taken in these edits (change them here if you disagree)

1. The **title stays**. "Adaptive forecasting from heterogeneous evidence" still describes the method.
2. The **framing moves from "fusion improves accuracy" to "a protocol that tests each evidence channel and recomposes the forecast from those that survive"**. Per SUMMARY.md, there is no accuracy claim over the best single model.
3. **Primary numbers:** backtest test cells without the reserved-window targets, plus the holdout. The smoothed cut is primary, and the raw cut appears in the main text.
4. **Event and intrinsic channels:** kept in the framework and reported as tested channels that calibrated to null. They are retired from the housing forecast path.
5. **Statistics** are cited properly. New bibliography entries are listed at the end of this file.

---

## E1. Abstract

**Current:** the whole abstract (`Real-world forecasting often requires ...` to `... in multi-horizon forecasting.`)

**Action:** replace the whole abstract. It keeps the authors' first four sentences almost unchanged, corrects the calibration sentence, and adds the results.

```latex
\begin{abstract}
\noindent
Real-world forecasting often requires integrating heterogeneous evidence whose predictive value varies across horizons, operating regimes, and domains. We introduce FACTS-MAS, a domain-agnostic multi-agent forecasting framework that decomposes evidence into specialized channels for temporal dynamics, exogenous context, seasonality, structural information, and discrete events. Each agent generates an independently testable forecast through a common interface, while leakage-safe historical validation measures its predictive skill relative to a naive reference. These skill estimates are used to weight and combine agent forecasts, so that uninformative evidence is downweighted rather than forced to participate. A calibration procedure selects domain-specific constants on historical validation folds and may set a constant to zero when a nonzero effect is not supported. We evaluate FACTS-MAS on weekly for-sale housing inventory for 15 U.S.\ metropolitan areas at horizons of 4, 8, and 13 weeks. Relative to a last-value forecast, the frozen configuration reduces mean absolute percentage error by 50\%, 50\%, and 47\% on Zillow's smoothed inventory series and by 24\%, 35\%, and 37\% on the unsmoothed series, and it retains these gains on a pre-registered holdout of observations after February 21, 2026 that were not accessed until the configuration was frozen. It also outperforms the TimesFM foundation model. However, a factor-adjusted damped exponential smoothing model performs as well as the fused forecast, and two of the five housing channels calibrate to a null effect. The evidence therefore supports FACTS-MAS as a systematic protocol for testing, calibrating, and recomposing evidence channels, rather than as a source of accuracy beyond the strongest single model.
\end{abstract}
```

[backtest: phase4_ci_CLEANBLEED.csv, phase5_raw_stats_levels.csv; holdout: phase7_holdout_mape.csv]

---

## E2. Introduction, paragraph 3 ("In this paper, we develop a new approach ...")

**E2a. Current:** `Domain-specific constants---lags, impact scales, and durations---are chosen by the same validation test, and these constants can be set to zero.`

**Action:** replace. The macro lags were pre-specified, not selected.

```latex
Domain-specific constants---event-impact scales, event durations, and
reversion caps---are chosen by the same validation test, and these constants
can be set to zero.
```

---

## E3. Introduction, paragraph 4 (the results paragraph, "We evaluate FACTS-MAS on the forecast of weekly U.S. housing inventory ...")

**Action:** replace the paragraph from `We evaluate FACTS-MAS on the forecast of weekly U.S.\ housing inventory` to `to assess the generalizability of the approach.`

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
model \citep{das2024timesfm} in the backtest, and on the holdout it is more
accurate than both TimesFM and a chain-of-thought language-model forecast
built on the NEXUS prompt \citep{das2026nexus}. It is not more accurate than a
factor-adjusted damped exponential smoothing model, with which it is
statistically equivalent at 8 weeks. We also transfer the method to weekly
unemployment insurance claims to assess the generalizability of the approach.
```

[backtest: phase4_ci_CLEANBLEED.csv; holdout: phase7_holdout_mape.csv; raw: phase5_raw_stats_levels.csv; TimesFM: phase6_timesfm_stats.csv (p <= 0.006)]

---

## E4. Contributions list

**E4a. Item 2. Current:** `The weights may depend on forecast horizon and operating regime when the validation sample is sufficient to support that conditioning.`

**Action:** replace with

```latex
The weights may depend on forecast horizon and operating regime when the
validation sample is sufficient to support that conditioning; a regime is
used only when it contains at least 13 distinct validation weeks, and
otherwise the pooled weights apply.
```

**E4b. Item 3. Current:** `including lags, impact scales, and event durations.`

**Action:** replace with `including event-impact scales, event durations, and reversion caps.`

**E4c. Item 4.** Replace the whole item:

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

---

## E5. Introduction, roadmap paragraph ("The remainder of the paper is organized as follows.")

**Current:** `Section~\ref{sec:discussion} discusses the housing findings and describes transfer of the method to other forecasting tasks, including unemployment.`

**Action:** replace with

```latex
Section~\ref{sec:discussion} discusses the housing findings, including the
holdout results, and reports the transfer of the method to weekly
unemployment insurance claims.
```

---

## E6. Related Work, 2.2, last sentences

**Current:** `The housing study uses Holt--Winters on the temporal channel and TimesFM~2.5, a later checkpoint of the TimesFM line, as a numerical baseline.`

**Action:** replace with

```latex
The housing study uses Holt's linear-trend exponential smoothing on the
temporal channel at the 4-week horizon and a factor-adjusted damped-trend
model \citep{gardner1985damped} at 8 and 13 weeks, and it uses TimesFM~2.5,
a later checkpoint of the TimesFM line, as a numerical baseline.
```

---

## E7. Related Work, 2.3, end of the first paragraph

**Current:** `\citet{timmermann2006combinations} reviews when such combination improves on the best individual forecast.`

**Action:** keep this sentence and add one after it. It sets up your main finding.

```latex
\citet{timmermann2006combinations} reviews when such combination improves on
the best individual forecast. That question is central here: a combination
of channels is only useful if it improves on the strongest single model that
could have been selected on the same validation data.
```

---

## E8. Related Work, 2.4 (Calibration and NEXUS)

**E8a. Current:** `Efficiency extensions of that design, including early-exit gating, spillover analysis, statistical output constraints, and batched reasoning, retain language-model reasoning on the main forecasting path.`

**Action:** replace with the following. These four modifications were removed from this paper (mismatch item B9).

```latex
Efficiency extensions of that design, such as early-exit gating and batched
reasoning, retain language-model reasoning on the main forecasting path; they
are not part of FACTS-MAS and are not evaluated here.
```

**E8b. Current:** `At each forecast origin, the fusion weights are updated from the channels' historical skill relative to a simple reference. Those weights may also depend on the forecast horizon and the operating regime when the validation sample is sufficient, so the combined forecast adapts to the channels that have recently proved useful.`

**Action:** replace. The weights are estimated per fold, not per origin; this is a new mismatch item, B17.

```latex
The fusion weights are estimated from the channels' historical skill relative
to a simple reference on a validation window that ends at the training cutoff,
and they are re-estimated whenever the training window is extended. Those
weights may also depend on the forecast horizon and the operating regime when
the validation sample is sufficient.
```

**E8c. Current:** `The housing experiments in Section~\ref{sec:housing} compare FACTS-MAS with a NEXUS chain-of-thought baseline and with a time-series foundation model.`

**Action:** replace with

```latex
The housing experiments in Section~\ref{sec:housing} compare FACTS-MAS with
a chain-of-thought language-model forecast that uses the NEXUS prompt, with a
time-series foundation model, and with strong statistical references.
```

---

## E9. Section 3 opening, the paragraph after eq. (3)

**Current:** `These maps determine the forecast. At each forecast origin, domain evidence is assigned to the $K$ channels, and each agent returns an independent forecast. A validation step scores these forecasts against a reference, a calibration step sets the channel weights, and, finally, adaptive fusion combines them into $\hat{\mathbf{y}}_{\mathrm{FACTS}}$.`

**Action:** replace. Calibration does not set the weights; the skill scores do (mismatch B8).

```latex
These maps determine the forecast. At each forecast origin, domain evidence is
assigned to the $K$ channels, and each agent returns an independent forecast.
A validation step scores each agent's historical forecasts against a
reference, the resulting skill scores set the fusion weights, and adaptive
fusion combines the agent forecasts into $\hat{\mathbf{y}}_{\mathrm{FACTS}}$.
Calibration is a separate, offline step that selects channel constants on
historical validation folds before the test period begins.
```

**E9b. Figure 1 caption. Current:** `Skill relative to a naive reference sets the fusion weights, which are recomputed when a new origin is reached.`

**Action:** replace with `Skill relative to a naive reference, measured on a validation window that ends at the training cutoff, sets the fusion weights.`

**Also:** if the figure image draws calibration inside the per-origin loop, redraw it so that calibration sits outside the loop, feeding constants into the channels (mismatch B8).

---

## E10. Section 3.4 (Skill-Based Adaptive Fusion)

**E10a. After eq. (softmax)** `where $\tau>0$ controls how strongly differences in historical skill affect the final allocation.` **add:**

```latex
In the housing configuration $\tau=0.05$. This value was set by hand from the
magnitude of validation skill scores, not selected by a search; a
validation-selected temperature was tested and not adopted
(Section~\ref{sec:housing}).
```

**E10b. Current (last paragraph):** `Weights are recomputed from rolling historical evidence rather than fixed once for the entire evaluation period. Consequently, $w_{k,t_1}^{(H)} \neq w_{k,t_2}^{(H)}$ is permitted as new validation evidence becomes available.`

**Action:** replace with

```latex
Weights are re-estimated whenever the training window is extended rather than
fixed once for the entire evaluation period. In the backtest, each fold
estimates its weights from a 104-week validation window ending at the training
cutoff and holds them fixed over that fold's test window. Consequently,
$w_{k,t_1}^{(H)} \neq w_{k,t_2}^{(H)}$ is permitted across folds, and within
a fold when $t_1$ and $t_2$ fall in different regimes.
```

---

## E11. Section 3.5 (Adaptive Context and Regime Conditioning), end of the second paragraph

**Current:** `FACTS-MAS allows the level of adaptation to be constrained by historical support rather than assuming that finer conditioning always improves generalization.`

**Action:** keep it and add the rule that was actually used (mismatch B7):

```latex
FACTS-MAS allows the level of adaptation to be constrained by historical
support rather than assuming that finer conditioning always improves
generalization. In the housing configuration, a regime-specific weight vector
is used only when the regime contains at least 13 distinct validation weeks;
otherwise the forecast falls back to the pooled weights. An earlier version
of the implementation had no such rule and assigned equal weights to every
channel when a regime had no validation data, which affected 780 test cells
and inflated the apparent benefit of regime conditioning.
```

---

## E12. Section 3.6 (Calibration)

**Current:** the paragraph ending `... can remain available even when its direct numerical adjustment is removed.`

**Action:** add a new paragraph after it (discloses F0.9):

```latex
In the housing study, each constant was selected once using the validation
windows of all six folds, with a candidate promoted only if it improved
validation loss in at least five of the six folds. Because later folds'
validation windows postdate earlier folds' test windows, this selection is not
fully nested in time. The only adopted calibration result, a zero event-impact
scale, changes test accuracy by at most 0.002 percentage points.
```

**Also fix:** the displayed equation $L_f(\theta)=\ldots$ ends with a comma followed by a blank line. Change the comma to a period.

---

## E13. Section 3.7 (Factor Attribution)

**Current:** eq. $C_{k,t}^{(H)} = w_{k,t}^{(H)}\hat{\mathbf{y}}_{k,t}^{(H)}$ and the sentence after it.

**Action:** replace the equation and add a short paragraph (mismatch B5; change-based attribution is primary):

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

---

## E14. Section 4 opening ("Three cases---weekly U.S. housing inventory ...")

**Current:** `Three cases---weekly U.S.\ housing inventory time series, energy demand time series, and unemployment time series---are analyzed by FACTS-MAS.` ... `Energy demand and unemployment are specified as transfer configurations for the third.`

**Action:** replace these two sentences (mismatch E1):

```latex
Two cases---weekly U.S.\ housing inventory and weekly unemployment insurance
claims---are analyzed by FACTS-MAS, and a third, energy demand, is specified
as a planned configuration only.
```

and

```latex
Unemployment claims are used for the third, and the energy configuration is
reported as a channel map without results.
```

---

## E15. Section 4.1, first paragraph (the protocol)

**Current:** `so fitting, calibration, lags, fusion weights, and the choice of weighting rule use only pre-test information.`

**Action:** replace the end of that sentence and add a disclosure (mismatch C1, C2, C3):

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

**Current:** `The embargo period takes at least the longest horizon, $H_{\max}=\max\{H_1,\ldots,H_q\}$; for weekly housing data it is 13 weeks.`

**Action:** add after it:

```latex
Validation forecasts used to estimate fusion weights are restricted to
targets dated on or before the training cutoff, so that no validation target
falls inside the embargo. Test targets dated on or after January 1, 2026, the
start of the reserved period, are excluded from the backtest; that period is
used only for the holdout described below.
```

---

## E16. Section 4.1, metrics paragraph, after the MAPE/RMSE equation

**Action:** add a statistics paragraph after the equation:

```latex
Accuracy differences are tested on the weekly cross-section of losses. For
each pair of methods, the panel Diebold--Mariano test of equal average
accuracy \citep{diebold1995comparing,qu2024comparing} uses a Newey--West
variance with lag $H-1$ \citep{newey1987simple}; per-MSA tests use the
small-sample correction of \citet{harvey1997testing} with the
Benjamini--Hochberg adjustment across the 15 MSAs
\citep{benjamini1995controlling}. Confidence intervals come from a moving
block bootstrap over whole weeks \citep{kunsch1989jackknife} with 5,000
replications and block length 13. Two methods are called equivalent when the
90\% confidence interval of their MAPE difference lies within $\pm 2\%$ of
the reference method's MAPE. The model confidence set \citep{hansen2011mcs}
summarizes the comparison among all methods.
```

---

## E17. Section 4.1, ablation paragraph

**Current:** `A positive value means that the channel contributes inside the combination.`

**Action:** add a definition after it (mismatch F2.8):

```latex
Removing a channel renormalizes the remaining weights in each cell without
re-estimating them. Removing adaptive weighting replaces every cell's weights
with the fold's pooled weights, which removes both horizon and regime
conditioning, so this ablation alone cannot separate the two.
```

---

## E18. Section 4.1, weighting-rules paragraph ("Four weighting rules are compared ...")

**Action:** replace the paragraph. The granular rule was chosen using test data and has been removed from the paper (A4).

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

---

## E19. Section 4.1, baselines paragraph ("Every compared method is scored ...")

**Action:** replace the whole paragraph (mismatch D1, C25):

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

---

## E20. Section 4.2 (housing), opening paragraph

**Current:** `It can populate all five channels in Section~\ref{sec:evidence_channels}: ... disaster declarations.`

**Action:** add after that sentence:

```latex
Two of these channels, the intrinsic and event channels, calibrate to a null
quantitative effect and are retired from the final housing forecast; they are
reported as tested channels.
```

---

## E21. 4.2.1 Domain Instantiation, data paragraph

**Current:** `Zillow's available weekly history begins on February 3, 2018 and extends through June 20, 2026 in the dataset used for this study. The resulting target panel contains 438 weekly observations per metropolitan statistical area (MSA) and 6,570 MSA--week observations overall.`

**Action:** replace (mismatch B12):

```latex
The target is Zillow's weekly for-sale inventory series, which Zillow reports
as a four-week trailing average; the unsmoothed weekly series is used as a
robustness check. The panel begins on February 3, 2018. The backtest uses
observations through February 21, 2026. The holdout uses a fresh download
taken on October 20, 2026 with observations through October 10, 2026 (454
weekly observations per metropolitan statistical area (MSA) and 6,810
MSA--week observations); every value dated on or before February 21, 2026 in
that download equals the earlier vintage, so no revision affects the results.
```

**Current:** `Nine target observations were missing in Zillow's source series: ...`

**Action:** keep the sentence as is. Add `The unsmoothed series has 20 missing observations, filled the same way.` and, after `... retained in a separate audit log.`, add `No missing observation falls in a test or holdout target.`

**Current:** `The decomposition is deliberately non-overlapping where possible. In particular, the temporal agent does not model seasonality internally because seasonality is represented by a separate evidence channel. This makes the incremental value of each component easier to evaluate and prevents two agents from encoding the same seasonal mechanism by construction.`

**Action:** replace:

```latex
The initial decomposition is deliberately non-overlapping: the 4-week
temporal agent does not model seasonality internally because seasonality is
represented by a separate evidence channel, which makes the incremental value
of each component easier to evaluate. At 8 and 13 weeks, however, the
temporal agent that survived validation is a factor-adjusted model that
reuses the seasonal channel's factors (see the temporal channel below), so
the two channels are not independent at those horizons.
```

---

## E22. Table 2 (housing channels)

**Action:** replace the table body rows:

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
Lagged ridge-regression modifier (pre-specified lags)
\\

Seasonal
&
Week-of-year inventory pattern
&
Ratio-to-moving-average seasonal factors (M4 Naive2 form)
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

---

## E23. Temporal channel paragraph

**Action:** replace the paragraph from `The temporal channel models the endogenous trajectory` to `... rather than manually introducing test-informed model overrides.` (mismatch A8: the old 2.73 / 5.51 / 7.79 numbers came from a superseded single-origin script).

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
consults test-fold accuracy, the holdout in Section~\ref{sec:discussion}
provides the clean test of the resulting configuration. Damping was chosen over
an undamped trend on validation data in every fold. Both models are fitted
separately for each MSA and forecast origin using only the available
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

[phase3_comparison_CLEANBLEED.csv: Holt, FA-ETS-d, naive; MSA wins vs naive 15 / 14 / 11; Holt worse than naive in folds 3-6 at 13wk (folds 3, 4, 5, 6 = four folds)]

---

## E24. Exogenous channel paragraph

**E24a. Current:** `The initial lags were 6 weeks for the mortgage rate, 10 weeks for the federal funds rate, 10 weeks for CPI, and 5 weeks for unemployment.`

**Action:** replace (mismatch B2): `The lags are pre-specified from the literature rather than fitted: 6 weeks for the mortgage rate, 10 weeks for the federal funds rate, 10 weeks for CPI, and 5 weeks for unemployment.`

**E24b. Current:** from `Mortgage rate, the theoretically most direct housing-finance variable,` to `... and only federal funds crossed the conventional 5\% threshold ($p=0.042$).`

**Action:** replace (mismatch A7):

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

[phase4_macro_specs.csv; phase3_comparison_CLEANBLEED.csv]

---

## E25. Seasonal channel paragraph

**Current:** `The week-of-year profile is smoothed circularly so that the transition between week 53 and week 1 is treated as adjacent rather than as a boundary. Factors are then normalized to have mean one.`

**Action:** replace (mismatch B11):

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

**Current:** `This formulation captures the annual shape of housing listings without requiring the temporal agent to model the same mechanism.`

**Action:** replace with

```latex
This forecast is the seasonally adjusted naive method of the M4 competition
\citep{makridakis2020m4} with leakage-safe factors. On its own it has MAPE
2.890\%, 4.774\%, and 6.945\% at 4, 8, and 13 weeks.
```

---

## E26. Intrinsic channel paragraph

**Current:** `with $\rho_{\max}=0.30$ in the current housing configuration. The quadratic schedule produces weak short-horizon pull and stronger structural reversion later in the forecast window.`

**Action:** replace (mismatch B3, B4, C12, C13; vintage disclosure F1.5):

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

[rho_max_calibration.csv (bounded re-run); F2.1; F1.5]

---

## E27. Event channel paragraph (the part received)

**E27a. Current:** `The Event Agent is the only housing component that uses an LLM.`

**Action:** replace with `The Event Agent is the only housing component that uses an LLM (gpt-4o-mini).` (mismatch B10)

**E27b.** The rest of this paragraph was cut off. In Part 2 it will:
- report that the calibrated impact scale is zero, using the bounded sweep values in place of Table 10's;
- state that the event output therefore equals the naive forecast in every cell;
- mark the channel retired.

---

## E28. New bibliography entries (add to your .bib)

```bibtex
@article{gardner1985damped,
  author  = {Gardner, Everette S. and McKenzie, Ed},
  title   = {Forecasting Trends in Time Series},
  journal = {Management Science},
  volume  = {31},
  number  = {10},
  pages   = {1237--1246},
  year    = {1985}
}
@article{assimakopoulos2000theta,
  author  = {Assimakopoulos, V. and Nikolopoulos, K.},
  title   = {The Theta Model: A Decomposition Approach to Forecasting},
  journal = {International Journal of Forecasting},
  volume  = {16},
  number  = {4},
  pages   = {521--530},
  year    = {2000}
}
@article{diebold1995comparing,
  author  = {Diebold, Francis X. and Mariano, Roberto S.},
  title   = {Comparing Predictive Accuracy},
  journal = {Journal of Business \& Economic Statistics},
  volume  = {13},
  number  = {3},
  pages   = {253--263},
  year    = {1995}
}
@article{harvey1997testing,
  author  = {Harvey, David and Leybourne, Stephen and Newbold, Paul},
  title   = {Testing the Equality of Prediction Mean Squared Errors},
  journal = {International Journal of Forecasting},
  volume  = {13},
  number  = {2},
  pages   = {281--291},
  year    = {1997}
}
@article{newey1987simple,
  author  = {Newey, Whitney K. and West, Kenneth D.},
  title   = {A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix},
  journal = {Econometrica},
  volume  = {55},
  number  = {3},
  pages   = {703--708},
  year    = {1987}
}
@article{benjamini1995controlling,
  author  = {Benjamini, Yoav and Hochberg, Yosef},
  title   = {Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {57},
  number  = {1},
  pages   = {289--300},
  year    = {1995}
}
@article{kunsch1989jackknife,
  author  = {K{\"u}nsch, Hans R.},
  title   = {The Jackknife and the Bootstrap for General Stationary Observations},
  journal = {The Annals of Statistics},
  volume  = {17},
  number  = {3},
  pages   = {1217--1241},
  year    = {1989}
}
@article{hansen2011mcs,
  author  = {Hansen, Peter R. and Lunde, Asger and Nason, James M.},
  title   = {The Model Confidence Set},
  journal = {Econometrica},
  volume  = {79},
  number  = {2},
  pages   = {453--497},
  year    = {2011}
}
@article{driscoll1998consistent,
  author  = {Driscoll, John C. and Kraay, Aart C.},
  title   = {Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data},
  journal = {Review of Economics and Statistics},
  volume  = {80},
  number  = {4},
  pages   = {549--560},
  year    = {1998}
}
@article{qu2024comparing,
  author  = {Qu, Ritong and Timmermann, Allan and Zhu, Yinchu},
  title   = {Comparing Forecasting Performance with Panel Data},
  journal = {International Journal of Forecasting},
  volume  = {40},
  number  = {3},
  pages   = {918--941},
  year    = {2024}
}
```

**Check before posting:** I am confident of the first nine entries. For `qu2024comparing`, check the volume and page numbers against the journal page.

## New mismatch item found while writing these edits

| # | Location | Paper says | Code says | Change |
|---|---|---|---|---|
| B17 | Sec. 2.4, Sec. 3 opening, Fig. 1 caption, Sec. 3.4 | Weights are updated or recomputed at each forecast origin | Weights are estimated once per fold from the 104-week validation window ending at `train_end` (`facts_mas/fusion_baseline.py:335`, `compute_fold_weights`) and held fixed over the fold's test window; only the regime cell used can vary by origin | Edits E8b, E9b, E10b |
