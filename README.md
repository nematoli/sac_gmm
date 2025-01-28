# SAC-_N_-GMM
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>SAC-_N_-GMM: Robot Skill Refining and Sequencing for Long-Horizon Manipulation Tasks</b>](https://akshaychandra.com/assets/pdf/masterproject-report.pdf)

[Akshay L Chandra](https://akshaychandra.com), [Iman Nematollahi](https://imanema.com/), [Tim Welschehold](https://rl.uni-freiburg.de/people/welschehold)

We present **SAC-_N_-GMM**, a single agent that learns to refine and sequence several robot skills to complete tasks. 

## Installation
To begin, clone this repository locally
```bash
git clone https://github.com/acl21/sac_n_gmm.git
export SACNGMM_ROOT=$(pwd)/sac_n_gmm

```
Install requirements:
```bash
cd SACNGMM_ROOT
conda create -n sacngmm_venv python=3.8
conda activate sacngmm_venv
sh install.sh
```


For Development:
```bash
pip install -r requirements-dev.txt
pre-commit install
```


## Download
Download the [CALVIN dataset](https://github.com/mees/calvin) and place it inside [dataset/](./dataset/). 

## Robot Skill Repertoire

### Step 1: Extract skill demos from the CALVIN dataset
Configure [config/demos.yaml](./config/demos.yaml).
```
> python sac_n_gmm/extract_demos.py skill='open_drawer'
```

### Step 2: Train and evaluate skill libraries (Dynamical Systems) with Riepybdlib 
Configure [config/gmm_train.yaml](./config/gmm_train.yaml).
```
> python sac_n_gmm/scripts/gmm_train.py skill='open_drawer'
```

Configure [config/gmm_eval.yaml](./config/gmm_eval.yaml).
```
> python sac_n_gmm/scripts/gmm_eval.py skill='open_drawer'
```

## Train RL Agent 
```
python sac_n_gmm/scripts/sac_n_gmm_train.py
```

## Citation

If you find the code useful, please cite:

**SAC-_N_-GMM**
```bibtex
@inproceedings{nematollahi22icra,
    author  = {Akshay L Chandra and Iman Nematollahi and Tim Welschehold},
    title   = {SAC-N -GMM: Robot Skill Refining and Sequencing for Long-Horizon Manipulation Tasks},
    booktitle = {Master's Project},
    journal={Robot Learning Lab, Freiburg}
    year = 2024,
    url={https://akshaychandra.com/assets/pdf/masterproject-report.pdf},
}
```

## License

MIT License