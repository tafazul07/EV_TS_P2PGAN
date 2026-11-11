"""Inventory trip files, schema variation, missing values, intervals and ranges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import list_trip_files, read_trip


def inspect_dataset(data_dir: str | Path, report_dir: str | Path) -> dict:
    files = list_trip_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No TripAxx/TripBxx CSVs found in {data_dir}")
    trip_rows, column_rows = [], []
    for path in files:
        df = read_trip(path)
        raw_columns = [c for c in df.columns if c != "trip_id"]
        times = pd.to_numeric(df["Time [s]"], errors="coerce").dropna().to_numpy()
        deltas = np.diff(times)
        interval = float(np.median(deltas)) if deltas.size else np.nan
        duplicate_rows = int(df.duplicated(subset=raw_columns).sum())
        trip_rows.append({
            "trip_id": path.stem,
            "season": "summer" if path.stem.startswith("TripA") else "winter",
            "records": len(df),
            "columns": len(raw_columns),
            "duplicate_measurement_rows": duplicate_rows,
            "median_sample_interval_s": interval,
            "column_names": " | ".join(raw_columns),
        })
        for column in raw_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            column_rows.append({
                "trip_id": path.stem,
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "min": values.min(skipna=True),
                "max": values.max(skipna=True),
                "mean": values.mean(skipna=True),
                "std": values.std(skipna=True),
                "duplicate_measurement_rows": duplicate_rows,
            })
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    trips = pd.DataFrame(trip_rows)
    columns = pd.DataFrame(column_rows)
    trips.to_csv(root / "trip_inventory.csv", index=False)
    columns.to_csv(root / "column_inventory.csv", index=False)
    summary = {
        "trip_count": int(len(trips)),
        "summer_trip_count": int((trips.season == "summer").sum()),
        "winter_trip_count": int((trips.season == "winter").sum()),
        "record_count": int(trips.records.sum()),
        "schema_widths": sorted(map(int, trips["columns"].unique())),
        "unique_header_sets": int(trips.column_names.nunique()),
        "sample_interval_modes_s": trips.median_sample_interval_s.value_counts().head(5).to_dict(),
        "duplicate_measurement_rows": int(trips.duplicate_measurement_rows.sum()),
    }
    (root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/extracted")
    parser.add_argument("--output-dir", default="results/inspection")
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

