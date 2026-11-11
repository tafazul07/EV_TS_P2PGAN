# Phase 1 — Paper and dataset inspection

**Status:** The assigned paper has been read in full. The paper-grounded feature mapping, dataset inventory, preprocessing pipeline, and trip eligibility analysis are complete. The 20 summer files without both required heater channels are excluded; no measurements are fabricated.

## Source files

- `Base_paper.pdf` was copied from Downloads to the project root for reproducibility.
- `Measurement Data.zip` was extracted to `data/extracted/`; the source ZIP remains unchanged.
- The archive contains 70 trip CSVs (`TripA01`–`TripA32`, `TripB01`–`TripB38`), `Overview.xlsx`, and `readin.m`.
- The semicolon-delimited CSVs have 1,094,793 records total and a 0.1-second sampling interval. Trip sizes range from 3,545 to 56,102 rows. Schema widths are 22, 23, 24, 28, and 48 columns. All 38 winter files have 48 columns; the 32 summer files vary from 22 to 28 columns.
- CSVs use Windows-1252 encoding for degree symbols. TripB38 has one malformed header, `Velocity [km/h]]]`; the loader applies one explicit alias to the canonical velocity name. Other source channels are preserved, including similar HVAC temperature headers that differ by whitespace.
- `Overview.xlsx` supplies trip metadata. `readin.m` loads one selected CSV and includes commented examples, but neither specifies a different feature mapping.

The generated [trip inventory](../results/inspection/trip_inventory.csv) contains per-trip rows, data types, columns, sampling interval, and duplicate-row checks. The [column inventory](../results/inspection/column_inventory.csv) contains per-trip missing counts, numeric ranges, means, and standard deviations. The [inspection summary](../results/inspection/dataset_summary.json) records schema and trip counts.

## Paper and feature mapping

Paper: Shyr-Long Jeng, “Generative Adversarial Network for Synthesizing Multivariate Time-Series Data in Electric Vehicle Driving Scenarios,” *Sensors* (2025), 25, 749. The detailed [method summary](PAPER_METHOD_SUMMARY.md) records model settings and ambiguities.

Figure 1(a), p. 6, identifies the ten inputs. It explicitly labels the heating feature `Requested Heating Power [W]`, selected instead of CAN/LIN heating channels. It labels the elevation channel “Elevation change”; the dataset supplies `Elevation [m]` and the paper gives no formula, so using raw elevation is recorded as an interpretation. The four target columns match the supplied data directly.

The paper-grounded [feature mapping](../feature_mapping.json) records source names, units, conversions, and season coverage. Heater Voltage and Heater Current appear in only 12 of 32 summer files, while present in all 38 winter files. The other requested channels are present in all trips after the TripB38 velocity-header alias.

The SOC column contains 30,793 hyphen or blank missing readings. Four large trailing gaps occur in TripB07 (4,175 rows), B09 (20,541), B28 (1,625), and B34 (4,407). They are not interpolated; windows touching missing required values are excluded.

## Preprocessing and windows

The pipeline applies a trailing 10-point moving average within each trip, then keeps every tenth point to form 1-second-spaced samples. This decimation is an interpretation of the paper's Figure 1 axis, which labels time as “1 s/step”; the article's text describes the 10-point filter on 100 ms records but does not explicitly state a decimation step. No smoothing or window crosses a trip boundary.

Eligible trips are the 50 with all required columns (12 summer and 38 winter). With strict missing-window rejection, the resulting count is **62,562 valid windows**: 37,975 train, 11,779 validation, and 12,808 test. This is 24,974 fewer than the paper's 87,536, consistent with the 20 excluded summer trips. The count is reported as observed and is not forced to the paper's value. Splits are reproducible by trip and season (seed 42), preventing overlap leakage.

Separate X and Y MinMax scalers are fit using training trips only. Gaps remain missing through smoothing/scaling; only complete windows are retained.

## Reproduction limits

Using 50 complete-feature trips rather than the paper's 70-trip corpus is a documented deviation imposed by absent heater channels. The paper's numeric L1 coefficient is unspecified; the config exposes an explicit assumption. The paper also has generator/discriminator table dimension inconsistencies; implementation choices are recorded in [methodology](METHODOLOGY.md). PyTorch is not installed in the available runtime and package-network access is unavailable, so model smoke training has not run here.
