"""TimesFM 2.5 baseline (Google's open-weights time-series foundation model).
This is a comparison baseline, not one of our production agents -- it exists
to answer "how does our fused system compare to the numbers-only foundation
model the NEXUS paper itself benchmarks against," same role cot_baseline.py
plays for the paper's CoT approach.

Checkpoint: google/timesfm-2.5-200m-pytorch, loaded via the official `timesfm`
pip package's TimesFM_2p5_200M_torch class (its own DEFAULT_REPO_ID). This IS
the exact "TimesFM-2.5" checkpoint the NEXUS paper references -- confirmed via
feasibility check, no substitution needed (unlike the ETS stand-in used for
the Mod A gate's TimesFM reference). Runs on CPU; ~925MB one-time download,
cached to ~/.cache/huggingface after.

No text/event context: TimesFM is a pure numerical foundation model --
"text-blind" by design, per the paper's own description of it. This matches
that description exactly rather than working around it.

Point-in-time discipline: forecast_timesfm takes an already-truncated
historical_series, same convention as cot_baseline.py's forecast_cot_nexus --
the caller is responsible for cutting the series off at the correct forecast
origin; this function just reads the last HISTORY_WEEKS entries and does not
know about folds or train_end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import timesfm

HISTORY_WEEKS = 52
CHECKPOINT_REPO = "google/timesfm-2.5-200m-pytorch"

_model = None  # lazy singleton -- loading + compiling is the expensive part


def _get_model():
    global _model
    if _model is None:
        _model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT_REPO)
        _model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=False,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
    return _model


def forecast_timesfm(historical_series: pd.Series, horizon_weeks: int) -> list[float]:
    """
    Point forecast from TimesFM 2.5.

    historical_series must already be truncated by the caller to the correct
    point-in-time forecast origin (matches forecast_cot_nexus's calling
    convention exactly, so both baselines can be driven by the same harness
    loop). Uses the last HISTORY_WEEKS (52) observations as context, same
    window as cot_baseline.py.

    Clamps negative outputs to 0 -- inventory counts cannot be negative, and
    the model has no inherent guarantee against it despite infer_is_positive
    being set (that flag informs the model's own normalization, it is not a
    hard constraint on the output).
    """
    model = _get_model()
    window = historical_series.tail(HISTORY_WEEKS)
    context = window.values.astype(np.float64)
    point, _quantiles = model.forecast(horizon=horizon_weeks, inputs=[context])
    values = np.maximum(point[0], 0.0)
    return [float(v) for v in values]
