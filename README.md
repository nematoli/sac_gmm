# SAC-GMM
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>Robot Skill Adaptation via Soft Actor-Critic Gaussian Mixture Models</b>](http://ais.informatik.uni-freiburg.de/publications/papers/nematollahi22icra.pdf)

[Iman Nematollahi*](https://imanema.com/), 
[Erick Rosete Beas*](https://erickrosete.com/), 
[Adrian Röfer](https://rl.uni-freiburg.de/people/roefer), 
[Tim Welschehold](http://www2.informatik.uni-freiburg.de/~twelsche/),
[Abhinav Valada](https://rl.uni-freiburg.de/people/valada),
[Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

We present **SAC-GMM**, ...

## Installation
To begin, clone this repository locally
```bash
git clone https://github.com/nematoli/sac_gmm.git
export SACGMM_ROOT=$(pwd)/sac_gmm

```
Install requirements:
```bash
cd SACGMM_ROOT
conda create -n sacgmm_venv python=3.8
conda activate sacgmm_venv
sh install.sh
```


For Development:
```bash
pip install -r requirements-dev.txt
pre-commit install
```


## Download
Download the [CALVIN dataset](https://github.com/mees/calvin) and place it inside [dataset/](./dataset/). 

## Skill Library

### Step 1: Extract skill demos from the CALVIN dataset
Configure [config/demos.yaml](./config/demos.yaml).
```
> python sacgmm/extract_demos.py skill='open_drawer'
```

### Step 2: Train and evaluate skill libraries (Dynamical Systems) with ManifoldGMM 
Configure [config/gmm_train.yaml](./config/gmm_train.yaml).
```
> python sac_gmm/scripts/gmm_train.py skill='open_drawer'
```

Configure [config/gmm_eval.yaml](./config/gmm_eval.yaml).
```
> python sac_gmm/scripts/gmm_eval.py skill='open_drawer'
```

### Pre-trained Models
We provide our final models for ...
```bash
cd SACGMM_ROOT/checkpoints
sh download_model_weights.sh
```


## Training
```
python 
```

### Ablations
```
python 
```

## Evaluation
```
python 
```

## Citation

If you find the code useful, please cite:

**SAC-GMM**
```bibtex
@inproceedings{nematollahi22icra,
    author  = {Iman Nematollahi and Erick Rosete-Beas and Adrian Roefer and Tim Welschehold and Abhinav Valada and Wolfram Burgard},
    title   = {Robot Skill Adaptation via Soft Actor-Critic Gaussian Mixture Models},
    booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation  (ICRA)},
    pages={8651-8657},
    year = 2022,
    url={http://ais.informatik.uni-freiburg.de/publications/papers/nematollahi22icra.pdf},
    address = {Philadelphia, USA}
}
```

## License

MIT License