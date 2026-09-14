"""JSON serialization helpers for numpy / pandas objects."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def _is_nan(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_json_compatible(obj: Any) -> Any:
    """Recursively convert an object into JSON-serializable Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_json_compatible(item) for item in obj.tolist()]

    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()

    if isinstance(obj, pd.Series):
        series = obj.copy()
        if isinstance(series.index, pd.DatetimeIndex):
            series.index = series.index.astype(str)
        else:
            series.index = series.index.map(lambda x: str(x))
        return {str(k): to_json_compatible(v) for k, v in series.to_dict().items()}

    if isinstance(obj, pd.DataFrame):
        frame = obj.copy()
        if isinstance(frame.index, pd.DatetimeIndex):
            frame.index = frame.index.astype(str)
        else:
            frame.index = frame.index.map(lambda x: str(x))
        frame.columns = [str(c) for c in frame.columns]
        return to_json_compatible(frame.to_dict(orient="split"))

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if isinstance(key, (pd.Timestamp, datetime, date)):
                str_key = key.isoformat()
            else:
                str_key = str(key)
            result[str_key] = to_json_compatible(value)
        return result

    if isinstance(obj, (list, tuple, set)):
        return [to_json_compatible(item) for item in obj]

    if _is_nan(obj):
        return None

    # Drop non-serializable engine / model objects by describing them
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith(
        ("analysis.", "models.", "performance_optimizer")
    ):
        class_name = obj.__class__.__name__
        # Prefer summarizing known backtest engine fields
        if class_name == "BacktestEngine":
            summary = {
                "type": class_name,
                "initial_capital": getattr(obj, "initial_capital", None),
                "cash": getattr(obj, "cash", None),
                "trades": to_json_compatible(getattr(obj, "trades", [])),
            }
            return summary
        return {"type": class_name}

    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError, ValueError):
        return str(obj)


def dumps_json(obj: Any, **kwargs) -> str:
    """Serialize ``obj`` to a JSON string after making it compatible."""
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("indent", 2)
    return json.dumps(to_json_compatible(obj), **kwargs)
