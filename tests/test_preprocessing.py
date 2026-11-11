import numpy as np
import pandas as pd

from src.preprocessing import smooth_trip, split_trip_ids


def test_trip_split_is_reproducible_and_disjoint():
    ids = [*(f"TripA{i:02}" for i in range(1, 13)), *(f"TripB{i:02}" for i in range(1, 39))]
    a = split_trip_ids(ids, 17, 0.70, 0.15)
    b = split_trip_ids(ids, 17, 0.70, 0.15)
    assert a == b
    assert set(a["train"]).isdisjoint(a["validation"])
    assert set(a["train"]).isdisjoint(a["test"])
    assert set(a["validation"]).isdisjoint(a["test"])
    assert set().union(*(set(v) for v in a.values())) == set(ids)


def test_smoothing_keeps_missing_source_values_missing():
    frame = pd.DataFrame({"signal": np.arange(20, dtype=float)})
    frame.loc[12, "signal"] = np.nan
    smoothed = smooth_trip(frame, ["signal"], window=10)
    assert smoothed.loc[12, "signal"] != smoothed.loc[12, "signal"]
    assert np.isfinite(smoothed.loc[11, "signal"])

