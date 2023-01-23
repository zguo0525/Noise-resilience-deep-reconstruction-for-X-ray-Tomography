# Noise-resilience deep reconstruction for X-ray Tomography
====================

This repository contains the jupyter notebooks, python libraries, and information needed to reproduce the results and figures in the paper "Noise-resilient deep tomographic imaging" (https://preprints.opticaopen.org/articles/preprint/Noise-resilient_deep_tomographic_imaging/21931557).

The git repository contains all the code needed, however there is also a substantial amount of data associated with this work which cannot be distributed via a git repository. The data can be found by request to the author. 

Organization
------------
- The FBP_algo folder contains all the important functions to compute FBP reconstruction.
- The util folder contains all the important functions to train our neural network.
- Simulate_Projections.ipynb is used to generate simulated data the X-ray projections for the paper's simulation section.
- Generating-E2E-Simulation-Data.ipynb is used to generate simulated training data for End-to-End training.
- Sim_Results.ipynb is used to perform analysis and generate figures for the simulation section of the paper.
- Experimental_Results.ipynb is used to perform analysis and generate figures for the experimental section of the paper.
