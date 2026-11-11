# Methodology and implementation decisions

## Paper method

The project follows Jeng (2025), *Sensors*, 25, 749. The paper uses 10 condition channels and generates four target channels. The generator is a 1D encoder, nine residual blocks, and decoder. Two conditional discriminator scales provide fine and coarse temporal feedback. The paper applies a 1-second, 10-point moving average, makes overlapping 256-step sequences, normalizes to [-1, 1], and trains with Adam at 0.0006 for 800 epochs, batch size 256, decaying the rate to zero over the final 200 epochs.

The exact input mappings and their season-level availability are in `feature_mapping.json`. The paper figure explicitly identifies `Requested Heating Power [W]`. The paper labels its elevation input “Elevation change,” but neither the prose nor figure supplies a calculation formula; this implementation uses the directly observed `Elevation [m]` channel and marks that interpretation as an assumption.

## Dataset decisions

The supplied ZIP has 70 trips and 1,094,793 raw records at 100 ms. It has inconsistent schema widths. Heater Voltage and Heater Current are absent from 20 summer trips, and TripB38 has the malformed velocity header `Velocity [km/h]]]`. The loader canonicalizes only this obvious extra-bracket typo. The 20 summer files without both heater signals are excluded from the model dataset; the resulting 50-trip subset is 12 summer and 38 winter trips. No absent feature is filled with a proxy.

SOC gaps include long trailing runs in TripB07 (4,175 rows), B09 (20,541), B28 (1,625), and B34 (4,407), plus short terminal or isolated gaps in some other trips. The source uses hyphen placeholders for missing readings. These parse as missing. No interpolation is performed. Any sequence touching a missing required value is rejected.

The source CSV is Windows-1252 encoded, not UTF-8. The loader uses `cp1252`. Raw trips stay independent throughout smoothing and window generation.

## Smoothing, downsampling, and sequence count

The raw sampling cadence is 0.1 seconds. A trailing ten-sample moving average is applied independently per trip. Figure 1 in the paper labels the plotted axis “Time steps [1 s/step]”, so the pipeline then retains every tenth filtered row to obtain 1-second-spaced steps. This decimation is an interpretation of the paper's figure and is recorded in `config.yaml` as `downsample_factor: 10`.

With the 50 complete-feature trips, strict missing-window rejection, and one-second steps, preprocessing creates **62,562 valid windows**: 37,975 train, 11,779 validation, and 12,808 test windows. This is 24,974 fewer than the paper's 87,536. The difference is consistent with excluding 20 summer trips that do not include heater voltage/current; a further exact match is not forced. Splits are by trip, stratified by season, and reproducible from seed 42.

## Scaling and leakage control

Independent MinMax scalers are fitted for X and Y using only training trips. The same fitted scalers transform validation/test trips and later invert synthetic target channels. Scaling is per feature. Missing values remain missing through smoothing and scaling.

## Architecture details and paper ambiguities

### Generator

The implementation uses reflection padding of 3, a 10→64 k7 convolution, four stride-2 k3 downsampling blocks (64→128→256→512→1024), nine 1024-channel residual blocks, four stride-2 transposed convolutions (1024→512→256→128→64), reflection padding 3, k7 output convolution to four channels, and Tanh. Each residual convolution uses BatchNorm and ReLU as specified in the Table 2 layer notation, with an identity skip connection.

For residual blocks, the table simultaneously lists a separate one-sample pad and convolution padding 1, while its dimensions correspond to reflect-pad-then-valid-convolution. The implementation uses reflection padding 1 and valid k3 convolutions to preserve length. Table 2 does not list output padding for the transposed convolutions; PyTorch needs `output_padding=1` at k3,s2,p1 to double temporal length exactly. Both choices preserve the stated 10x256 to 4x256 shapes.

### Discriminator

The discriminator concatenates 10 input channels and four target channels. The fine branch applies k4 stride-2 convolutions to 64, 128, and 256 channels, then stride-1 convolutions to 512 and one output channel, followed by sigmoid. The coarse branch average-pools (k3,s2,p1) and runs the same channel progression.

Table 3 gives the coarse final convolution length as 19, but lists the following sigmoid length as 35. Sigmoid preserves sequence length; the implementation uses 19. The table's average-pool padding of 1 is followed so length is 128; the narrative's “no padding” conflicts with this.

## Loss and training assumptions

The paper defines conditional adversarial loss plus `lambda * L1` but does not state a numeric lambda. `config.yaml` exposes `lambda_l1` and uses 100 as a labeled implementation assumption. Training uses binary cross entropy over both sigmoid discriminator branches and a non-saturating generator target-real loss; this is the stable implementation form of the paper's minimax objective. This distinction is recorded rather than claimed as a verbatim loss implementation.

The paper says dropout introduces stochasticity but does not specify probability or placement. The implementation uses 0.5 dropout in each residual block, labeled as a pix2pix-style assumption. External Gaussian noise is not added.

## Evaluation semantics

Metrics compare each generated held-out window with its paired measured target under the same conditioning window. They are conditional reconstruction metrics over unseen trips, not future forecasts. RMSE and MAE are reported in each target's physical unit. DTW is mean absolute cumulative alignment cost per path step on a four-times-downsampled sequence for the first 100 held-out windows to limit computation. PCA and t-SNE compare flattened real and synthetic paired windows. All project results must be generated from this implementation; paper values are reference-only.

