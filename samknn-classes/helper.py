import numpy as np
from sklearn.utils.extmath import softmax as skl_softmax
from typing import Dict, Hashable
import datetime
from typing import Any

def _to_numeric(v: Any) -> float | None:
    """Convert numeric-like values to float. Return None if not numeric-like."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, datetime.datetime):
        return float(v.timestamp())
    if isinstance(v, datetime.date):
        # date -> midnight timestamp
        return float(datetime.datetime(v.year, v.month, v.day).timestamp())
    return None

def mixed_distance_dict(
    xa: dict,
    xb: dict,
    num_minmax: dict | None = None,
    sensitive_key: str | None = None,
    drop_sensitive: bool = True,
    categorical_features: set[str] | None = None,
    missing_cost: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Mixed-type distance (Gower-like):
      - numeric/datetime/date: normalized abs diff using running min/max (num_minmax)
      - categorical/other: 0 if equal else 1
      - missing: missing_cost (default 1)

    If you pass in `num_minmax`, it will be UPDATED in-place:
        num_minmax[feature] = {"min": ..., "max": ...}
    """
    keys = set(xa) | set(xb)
    if drop_sensitive and sensitive_key is not None:
        keys.discard(sensitive_key)

    if num_minmax is None:
        num_minmax = {}

    xa_get = xa.get
    xb_get = xb.get
    dist_sum = 0.0
    count = 0

    for k in keys:
        a = xa_get(k)
        b = xb_get(k)

        # missing
        if a is None or b is None:
            dist_sum += float(missing_cost)
            count += 1
            continue

        if categorical_features and k in categorical_features:
            dist_sum += 0.0 if a == b else 1.0
            count += 1
            continue

        a_num = _to_numeric(a)
        b_num = _to_numeric(b)

        # numeric/date/datetime
        if a_num is not None and b_num is not None:
            mm = num_minmax.get(k)
            if mm is None:
                mm = {"min": min(a_num, b_num), "max": max(a_num, b_num)}
                num_minmax[k] = mm
            else:
                if a_num < mm["min"]:
                    mm["min"] = a_num
                if b_num < mm["min"]:
                    mm["min"] = b_num
                if a_num > mm["max"]:
                    mm["max"] = a_num
                if b_num > mm["max"]:
                    mm["max"] = b_num

            rng = mm["max"] - mm["min"]

            if rng <= eps:
                d = 0.0
            else:
                d = abs(a_num - b_num) / (rng + eps)
            dist_sum += d
            count += 1
            continue
        # fallback for non-numeric, non-declared categorical
        dist_sum += 0.0 if a == b else 1.0
        count += 1


    return dist_sum / count if count else 0.0

def softmax(scores: Dict[Hashable, float]) -> Dict[Hashable, float]:
    if not scores:
        return {}

    keys = list(scores.keys())
    values = np.array([[scores[k] for k in keys]])

    probs = skl_softmax(values)[0]

    return {k: float(p) for k, p in zip(keys, probs)}


