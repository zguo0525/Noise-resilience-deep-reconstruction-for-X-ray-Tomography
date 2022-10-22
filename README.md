# Noise-resilience-deep-reconstruction-for-X-ray-Tomography
====================

This repository contains the jupyter notebooks, python libraries, and information needed to reproduce the results and figures in the paper "Randomized probe imaging through deep k-learning" (https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-2-2247&id=467928). (Zhen Guo, Abraham Levitan, George Barbastathis, and Riccardo Comin, "Randomized probe imaging through deep k-learning," Opt. Express 30, 2247-2264 (2022))

The git repository contains all the code needed, however there is also a substantial amount of data associated with this work which cannot be distributed via a git repository. The data can be found by request to the author. The code in this repository assumes that the data has been downloaded and located in a folder "data/" placed in the top-level directory within the repository. If the data is located elsewhere on your system, you can edit the data folder prefix at the top of each jupyter notebook.


Organization
------------
- The RPI_tools folder contains all the important functions to generative simulation data.
    - pytorch_tools contains the code used for iterative reconstructions
    - tf_tools contains code used for simulating RPI experiments in tensorflow
- The Deep-k-learning folder contains the designs for the non generative and generative deep k-learning architecutre, as well as scripts to use them using simulation and experimental data.
- Generating-Simulation-Data.ipynb is used to generate simulated data and approximants for the paper's simulation section.
- Generating-E2E-Simulation-Data.ipynb is used to generate simulated training data for End-to-End training.
- Process-Simulation-Data.ipynb is used to perform analysis and generate figures for the simulation section of the paper.
- Generating-Experimental-Data.ipynb is used to experimental data to generate approximants for the paper's simulation section.
- Process-Experimental-Data.ipynb is used to perform analysis and generate figures for the experimental section of the paper.
