# HaptoPFM-Haptotaxis-Phase-Field-Model

A phase field model with stochastic input that simulates cellular gradient sensing, morphodynamics, and fidelity of haptotaxis.

This repository contains the code used to run a phase field cell migration model that simulates signaling and morphodynamics in response to stochastic adhesion gradients. It was developed for the PLOS Computational Biology publication with the same title.

<img width="442" height="420" alt="image" src="https://github.com/user-attachments/assets/ee15c89d-8ed3-4d37-b16e-827fa0c9f5e7" /> 

<img width="442" height="420" alt="image" src="https://github.com/user-attachments/assets/2a3b3205-8e71-4298-933a-a1cd1b5d98c4" />

## Repository Information

- Repository: HaptoPFM-Haptotaxis-Phase-Field-Model
- Branch: main
- Platform: Linux-oriented workflows, including sample cluster submission scripts

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/hmapodac/HaptoPFM-Haptotaxis-Phase-Field-Model.git
cd HaptoPFM-Haptotaxis-Phase-Field-Model
conda create --name phasemodel1 -f environment.yml
```

## Usage

The model requires a few user-supplied inputs:

1. Random number sets that describe the adhesion networks. A code for generating these is stored in the adhesions folder, along with a sample random number set.
2. A parameters CSV file that provides the full parameter set for the model. A sample file is stored in the base folder.
3. An initialized phase field that allows the diffusion-like term to smooth the transition between the two phases of phi, where intracellular is 1 and extracellular is 0. A sample initialization and the code used to create it are supplied in the initialize folder.

The codes were run on a Linux cluster, and sample C shell submission scripts are included as `Submit1.csh` and `Submit2.csh`.

The main entry point is the `FipyRun#.py` code. It uses `concurrent.futures` to launch a process pool and run multiple modeling conditions across computational nodes. `FipyRun#.py` calls `PhaseModel#.py`, which gathers parameters, initializes the adhesion network, initializes the phase field variables, and starts the time-stepping loop.

Within the loop, the following codes are called in order:

1. `FipyModelTimestep.py` receives the previous timepoint data and simulates the phase field variables in the subdomain for one timestep, then saves timepoint-by-timepoint cell analyses.
2. `FracOcc.py` receives the new phase field data and the previous fractional occupancy and adhesion data, then simulates ECM removal or degradation when `k_rem` is not zero.
3. `Adhesions.py` is called every other loop using the random number sets and fractional occupancy to update the adhesion network in both the full domain and the subset within the phase field subdomain.
4. `Displacement.py` is called when the phase field cell centroid deviates 0.5 units in x or y from the subdomain center, recentering the cell by moving the subdomain.

After the loop finishes, movies of cell migration and the adhesion network are generated both within `PhaseModel#.py` and by calling `Analysis_VideoGeneration.py`. Saved data is organized and reformatted into compressed Parquet files using `ReformatFiles.py` and `ReformatFilesFracOcc.py`.

## Contributors

- Jason Haugh, jmhaugh@ncsu.edu
- Joseph Koelbl, jmkoelbl96@gmail.com
- Hector Apodaca, hmapodac@ncsu.edu
