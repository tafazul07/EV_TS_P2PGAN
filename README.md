# EV time-series synthesis with TS-p2pGAN

This project implements the conditional TS-p2pGAN approach described in `Base_paper.pdf` using the supplied BMW i3 trip measurements. The generator maps ten observed vehicle, battery, heating, and ambient signals to four target signals: SOC, battery voltage, motor torque, and longitudinal acceleration.

The paper's full 70-trip setup cannot be reproduced exactly from the supplied archive: 20 summer CSVs do not contain Heater Voltage or Heater Current. The pipeline excludes those 20 files and uses the 50 trips that contain every required channel. It does not create values for absent columns. TripB38 has an extra closing bracket in its velocity header; the loader applies and records a single header-name correction. SOC includes long missing stretches near the ends of several winter trips. Those samples and any window touching them are omitted rather than interpolated.

## What is implemented

- Dataset inventory, schema and missing-value analysis.
- Paper-grounded 10-input/4-output feature map.
- Trip-level train, validation, and test split; overlapping windows never cross trips or splits.
- Ten-sample moving average, then decimation to one-second steps. The one-second cadence is consistent with Figure 1's time axis; this is needed to reconcile sequence scale with the paper's preprocessing description.
- Train-only X/Y MinMax scaling to [-1, 1] with saved scalers.
- Nine-block 1D residual generator and two-scale conditional discriminator.
- GAN training, resumable checkpoints, loss logs, shape tests, and smoke mode.
- Held-out paired metrics, plots, and labeled synthetic CSV generation.

Architecture and data interpretation decisions are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md). Paper-reported metrics are not project results.

## Requirements

- Python 3.11 or later
- PyTorch 2.1 or later (CUDA recommended for full training; CPU works for smoke runs)
- Packages in `requirements.txt`

Create an environment and install dependencies from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`config.yaml` is written as JSON syntax, which is also valid YAML. The implementation reads it with Python's standard JSON parser to avoid an extra config-parser dependency.

## Dataset setup

The archive supplied with the assignment was extracted to `data/extracted/`. To re-extract on Windows:

```powershell
Expand-Archive -LiteralPath "Measurement Data.zip" -DestinationPath data/extracted
```

Keep each `Trip*.csv` separate. The loader reads the semicolon-delimited files as Windows-1252 so the degree symbols in temperature names are decoded correctly.

## Run the pipeline

Inspect all raw trips and regenerate the per-trip and per-column reports:

```powershell
python -m src.data_inspection --data-dir data/extracted --output-dir results/inspection
```

Prepare data, create the reproducible trip split, fit scalers using only training trips, and build per-trip normalized arrays:

```powershell
python -m src.preprocessing --config config.yaml
```

Run the one-epoch smoke mode (16 sequences, batch size 2):

```powershell
python -m src.train --smoke
```

Train using the paper's 800 epochs, batch size 256, learning rate 0.0006, and decay over the final 200 epochs:

```powershell
python -m src.train
```

Resume from the last checkpoint:

```powershell
python -m src.train --resume checkpoints/last.pt
```

Evaluate on held-out trips and save RMSE, MAE, DTW, PCA, t-SNE, waveforms, distributions, and correlation plots:

```powershell
python -m src.evaluate --checkpoint checkpoints/best.pt
```

Generate one synthetic 256-step sequence from a complete input trip:

```powershell
python -m src.generate data/extracted/TripB01.csv --checkpoint checkpoints/best.pt --sequences 1
```

Generated CSV rows carry `record_type=synthetic`, source trip, sequence number, and source row index. They must not be represented as measured records.

## Outputs

- `data/processed/`: normalized per-trip arrays, split IDs, scaler objects, and window counts.
- `results/logs/`: epoch-level generator/discriminator losses and learning rate.
- `checkpoints/`: last and best validation checkpoints.
- `results/metrics/`: held-out metrics, summaries, and target correlation matrices.
- `results/plots/`: held-out comparisons and embedding/distribution figures.
- `results/generated_data/`: synthetic CSV output.

## Important assumptions and limitations

- The paper's Figure 1 explicitly names `Requested Heating Power [W]`; CAN and LIN heating channels are not substituted.
- The paper says “Elevation change” but the supplied raw channel is `Elevation [m]`; no derivation is specified, so the raw source channel is used and this is documented as an interpretation.
- The paper does not give the numeric L1 weight. `lambda_l1=100` in `config.yaml` is an explicit configurable pix2pix-style implementation assumption, not a reported paper setting.
- The paper mentions dropout but gives neither probability nor exact placement. A 0.5 dropout inside each residual block is a configurable implementation assumption.
- 50 eligible trips and a trip-level holdout differ from the paper's 70-trip experiment. The resulting model has not been trained until a smoke run and experiment are completed; never cite paper table scores as this project's results.
- 100% reproducibility of paper metrics is not claimed. The paper does not describe a held-out trip split and has a reported sequence count that is inconsistent with using all original 100 ms rows as 256-step windows.

## Common issues

- **`No module named torch`:** install `requirements.txt` in a supported Python environment. This workspace's available Python runtimes did not include PyTorch, and network access prevented installing it here.
- **No valid windows:** run preprocessing first and inspect `data/processed/excluded_trips.json` and `preprocessing_summary.json`.
- **Missing conditioning columns:** choose a trip with all ten mapped inputs. Missing measurements are not fabricated or silently substituted.
- **CUDA out of memory:** set a smaller `training.batch_size` in `config.yaml`; record this as a deviation from the paper.

