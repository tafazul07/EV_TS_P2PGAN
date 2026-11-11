# Paper methodology notes (Phase 1)

Source inspected in full: Jeng, “Generative Adversarial Network for Synthesizing Multivariate Time-Series Data in Electric Vehicle Driving Scenarios,” *Sensors* 2025, 25, 749 (27 pages).

## Paper-specified dataset and features

The paper describes 70 BMW i3 trips, 32 summer Group A and 38 winter Group B, sampled every 100 ms. Figure 1(a), page 6, shows these ten generator conditions: Battery Current [A], Throttle [%], Velocity [km/h], Elevation change, Requested Heating Power [W], AirCon Power [kW], Heater Voltage [V], Heater Current [A], Battery Temperature [°C], Ambient Temperature [°C]. Figure 1(b) target channels are SoC [%], Battery Voltage [V], Motor Torque [Nm], and Longitudinal Acceleration [m/s^2]. The dataset headers directly match all target channels and eight of the ten input channels in every trip. Heater Voltage and Heater Current are present only in 12/32 summer files and all 38 winter files. No values were imputed.

The paper selects Requested Heating Power [W] in its plotted inputs, not `Heating Power CAN [kW]` or `Heating Power LIN [W]`; the mapping follows that explicit figure evidence. It plots “Elevation change,” while the provided file header is `Elevation [m]`; using the raw elevation channel is an evidence-based interpretation but still an assumption because no formula for a derived change channel is specified.

The source CSVs encode the degree sign as Windows-1252 byte `0xB0`; UTF-8 decoding displays `�C`. A data loader should decode the CSV as cp1252/Windows-1252 and then handle the actual header `Battery Temperature [°C]` and `Ambient Temperature [°C]`.

## Preprocessing reported by the paper

- Apply a 1-second, 10-point moving average at 100 ms sampling.
- Slice into overlapping 256-sample sequences with stride 1.
- Normalize inputs to [-1, 1]. The paper does not explain scaler fit scope or trip-level validation splitting.
- It reports 87,536 sequences. The count may not reproduce with the supplied exact files and the 14 required features. Missing heater columns in 20 summer trips prevent straightforward all-70-trip construction.

## Generator — Table 2, pages 9–10

Input shape is 10 x 256. The front end starts with reflection padding 3, then Conv1d blocks: 10→64 (k7,s1), 64→128 (k3,s2,p1), 128→256 (k3,s2,p1), 256→512 (k3,s2,p1), 512→1024 (k3,s2,p1). The paper defines Conv1dN+R as Conv1d + BatchNorm1d + ReLU. Nine 1024-channel residual blocks each use two k3,s1 convolutions with BatchNorm1d and ReLU plus identity skip connections. The back end has four transposed convolutions 1024→512→256→128→64, each k3,s2,p1, then reflection padding 3, a final 64→4 Conv1d k7,s1, and Tanh. The paper's stated dimensions are 256→128→64→32→16 then back to 256.

Table 2 is ambiguous about reflection padding inside residual blocks: its rows show a separate Pad1d of 1 and also list convolution padding 1, while its reported intermediate dimensions (16→18→16) correspond to reflection-pad-then-valid convolution. It also omits transposed-convolution `output_padding`; PyTorch requires output_padding=1 for k3,s2,p1 to double each length exactly. These decisions must be spelled out and shape-checked if implementation proceeds.

## Discriminators — Table 3, pages 10–11

Both branches receive the concatenation of ten condition and four target channels (14 x 256). D1 is the fine branch: Conv1d k4,s2,p2, channel sizes 64,128,256; then k4,s1,p2 to 512 and 1; Sigmoid. Activations are LeakyReLU; BatchNorm1d applies to the indicated N+R rows. D2 first average-pools with k3,s2,p1 (length 128), then repeats the branch with channels 64,128,256,512,1 and Sigmoid.

Table 3 has a dimension inconsistency: the D2 Conv_10 output is listed as length 19 but Sigmoid_2 is listed as length 35. Sigmoid does not change length, so 19 is the consistent shape. The narrative describes pooling with no padding, but the table specifies p=1 and reports length 128; implementation should follow the table's explicit (3,2,1) to achieve 128.

## Loss and training settings

The paper gives the conditional adversarial objective and an L1 reconstruction term weighted by lambda (Equation 4). It does not provide a numeric lambda in the inspected paper. It says external Gaussian noise is unnecessary and dropout is the source of stochasticity. Settings: Python 3.11.6, PyTorch 2.1, Adam, initial learning rate 0.0006, batch size 256, 800 epochs, linear/gradual decay to zero over the final 200 epochs; full dataset reshuffled each epoch. The training hardware reported is an RTX 4090 24 GB.

The paper does not describe a trip-level held-out test split. Its reported metrics must not be presented as independent held-out results for this implementation.

## Evaluation described

Metrics: RMSE, MAE, and DTW; qualitative comparisons include time-series plots and PCA/t-SNE. Reported comparisons include TS-p2pGAN, TTS-GAN, and TimeGAN. The paper reports performance values, but they are paper results and must not be presented as project experiment results.
