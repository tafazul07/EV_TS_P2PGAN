# Presentation outline

1. **Project title** — conditional synthetic EV time series with TS-p2pGAN.
2. **EV measurement data** — 70 BMW i3 trips, summer and winter, 100 ms sampling.
3. **Research problem** — paired multi-signal synthesis from limited observed trips.
4. **Objectives** — map ten conditions to four generated signals.
5. **Dataset inspection** — 1,094,793 rows, variable schemas, 32 summer and 38 winter.
6. **Feature selection** — list inputs and targets; requested heating power comes from Fig. 1(a).
7. **Data quality** — 20 summer trips lack heater voltage/current; long SOC gaps remain missing.
8. **Preprocessing** — moving average, decimation to 1 s/step, trip-safe split and scaling.
9. **Conditional GAN** — paired condition and target training idea.
10. **TS-p2pGAN architecture** — 1D encoder, nine residual blocks, decoder, two discriminator scales.
11. **Training settings** — Adam, 0.0006, batch 256, 800 epochs, final-200-epoch decay; lambda/dropout assumptions.
12. **Evaluation design** — held-out paired RMSE, MAE, DTW, PCA, t-SNE.
13. **Project results** — pending training; current preprocessing produced 62,562 windows.
14. **Limitations** — 50 trips available for all 10 inputs; missing runtime PyTorch.
15. **Conclusion and next steps** — run smoke test, train, evaluate, generate labeled CSV.

No paper-reported metrics are presented as project results. Results slide remains pending until training has completed.
