# Unveiling the Mechanisms of Bias
This repository corresponds to the Master's thesis in Artificial Intelligence by Tarmo Pungas, at University of Amsterdam, 2024. This research is part of a project on Bias Identification methods initiated by Rhite and has been guided and supervised by Rhite and the University of Amsterdam.

The repo is based on the [code](https://github.com/saprmarks/geometry-of-truth/tree/nnsight) from the paper <a href="https://arxiv.org/abs/2310.06824">*The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*</a> by Samuel Marks and Max Tegmark. We thank the authors for making their code publicly available.

**⚠️ Note on Code Quality**

Please note that this code was developed as part of a thesis project and, due to time constraints, has not undergone extensive optimization or formal quality checks. While it serves the primary purpose of supporting the research findings, it may not meet industry standards in terms of performance, maintainability, security or robustness.

Users are welcome to use and explore the code, but we recommend careful consideration and further testing before applying it in any production environment. Contributions for improvements or optimizations are also encouraged.

## Set-up
1. Navigate to the location that you want to clone this repo to, clone and enter the repo, and install the requirements.
```
git clone https://github.com/tarmopungas/msc-thesis.git
cd msc-thesis
pip install -r requirements.txt
```
2. Add any .csv datasets you would like to work with to the `datasets` folder. See `datasets/experiment_cps.csv` for how to format the files.
3. If you are using locally stored language models, specify the absolute path for the directory with model weights in `config.ini`. You can also use HuggingFace repos. 
4. Generate activations for the datasets you'd like to work with using a command like
```
python generate_acts.py --model llama-13b --layers 8 10 12 --datasets cities neg_cities --device cuda:0
```
These activations will be stored in the acts directory. If you want to save activations for all layers, simply use `--layers -1`.

Note that it is also possible to use [NNsight](https://nnsight.net/) to run inference remotely. To do this, join the NDIF Discord community and request an API key. You can then use `--device remote` when running any of the scripts.

## Files
This directory contains the following files:
* `acts`: the activations will be saved to this directory
* `data_processing`: StereoSet and CrowS-Pairs data, including processing scripts
* `datasets`: .csv files with labeled data
* `experimental_outputs`: the results will be saved to this directory
* `figures`: all the figures produced in the thesis
* `job_files`: example job files for running the scripts on SLURM
* `bias_patching.py`: script for running the patching experiment
* `config.ini`: specify which models to use here
* `dataexplorer.ipynb`: notebook for generating PCA visualizations
* `generalization.ipynb`: notebook for running the generalization experiment
* `generate_acts.py`: script for generating model activations
* `interventions`: script for running the intervention experiment
* `patching_nb.py` and `patching_nb.ipynb`: for creating a figure from the patching experiment results
* `patching_prompts.txt`: prompts used in the thesis for all patching experiments
* `probes.py`: definitions of logistic regression and mass-mean probes
* `uncertainties.py`: script for calculating uncertainties of the normalized indirect effect
* `utils.py` and `visualization_utils.py`: utilities for managing datasets and producing visualizations.
