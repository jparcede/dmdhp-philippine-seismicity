# DMDHP — 2-Zone Omori-Utsu Hawkes Process

Code for: Arcede et al. (2026). Temporal Kernel Choice Critically Affects
Depth-Dependent Aftershock Productivity Estimates.
Submitted to JGR: Solid Earth. DOI: 10.5281/zenodo.19699132

## Key Finding
Exponential kernel: R=7.25, p=0.0007, AIC=-4974
Omori-Utsu kernel: R=1.89, p=0.12,  AIC=-5048  (DELTA_AIC=74.2 better)
The exponential kernel inflates R — methodological warning for all
depth-stratified Hawkes process analyses.

## Primary Scripts
dmdhp_omori_full.py      — Primary Omori-Utsu analysis
dmdhp_omori_mc_v2.py     — Monte Carlo simulation
dmdhp_kernel_comparison.py — Kernel comparison
dmdhp_sensitivity.py     — Sensitivity analysis

## Results (Omori-Utsu kernel)
Hinatuan 2023:      R=1.894, p=0.121, c=0.0871, p_omori=0.473
Davao Oriental 2025: R=0.804, p=0.059
Davao del Sur 2019:  R=9.889, p=0.234
MC Power: 30-34% at R=2-5 (mean Nns~20 per simulation)

## Contact
jparcede@carsu.edu.ph — Caraga State University
