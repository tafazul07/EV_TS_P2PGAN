# Viva questions and answers

1. **What does TS-p2pGAN generate?** Four channels: SOC, battery voltage, motor torque, and longitudinal acceleration, conditioned on ten measured signals.
2. **Why use a conditional GAN?** It learns outputs paired with specific driving and environmental conditions instead of generating an unrelated sequence from noise alone.
3. **Why does the generator use residual blocks?** Identity skips allow deeper temporal feature processing while preserving a direct information path.
4. **Why are there two discriminators?** The full-resolution branch checks local details, while the pooled branch checks broader temporal patterns.
5. **Why are the discriminator inputs 14 channels?** They concatenate ten condition channels with four real or generated target channels.
6. **Why apply a moving average?** The paper uses a one-second ten-sample filter to reduce short-term measurement noise.
7. **Why downsample to one-second steps?** Figure 1 labels its temporal axis at one second per step. The paper text does not explicitly state decimation, so this is documented as an interpretation.
8. **Why split by trip rather than by windows?** Overlapping windows from a trip are highly similar; trip-level splits prevent those windows from leaking into evaluation.
9. **Why exclude 20 summer trips?** Their files do not contain required Heater Voltage and Heater Current channels. The values cannot be reconstructed from the supplied data without inventing them.
10. **Which heating-power signal is used?** Requested Heating Power [W], because that exact channel is labeled in Figure 1(a).
11. **Why not interpolate SOC?** Four trips have thousands of trailing missing SOC rows. Interpolation would manufacture large unmeasured trajectories.
12. **Why fit separate scalers?** Inputs and targets have different channels and physical ranges; independent scalers preserve their feature-wise mappings.
13. **What does the L1 term do?** It encourages generated outputs to remain close to the paired measured targets under the same conditions.
14. **What is lambda?** The paper leaves its value unspecified. The project exposes the coefficient and labels its initial value as an implementation assumption.
15. **What is dropout used for?** The paper says dropout introduces stochasticity in this conditional generator; its rate and placement are not specified, so the project documents its chosen assumption.
16. **What is DTW?** A sequence distance that permits nonlinear temporal alignment rather than requiring each sample to match at the identical index.
17. **Are evaluation metrics forecasts?** No. They compare outputs conditioned on held-out trip windows, so they are paired conditional reconstruction measures.
18. **Why does the count differ from 87,536?** The supplied archive lacks two paper inputs in 20 summer trips. The actual 50-trip pipeline yields 62,562 valid windows after missing target windows are removed.
19. **Did this project reach the paper's scores?** No project score exists until its own training and evaluation finish. Published values remain paper results only.
20. **What remains to complete the experiment?** Install PyTorch, run the shape tests and smoke mode, train with available compute, and then generate held-out metrics and plots.
