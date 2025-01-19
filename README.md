# Official PyTorch Implementation for “Doubly Robust Conditional VAE via Decoder Calibration: An Implicit KL Annealing Approach” 
<!-- ABOUT THE PROJECT -->

In this work, we introduce $\sigma$-calibration training techniques to achieve precise estimation of decoder variance while mitigating posterior collapse. Our Calibrated Robust CVAE incorporates a dynamic decoder variance, actively monitors the KLD component of the ELBO, and "calibrates" the decoder output variance at convergence when needed. This approach eliminates the need to reweight the ELBO loss function by $\beta$ hyperparameter for preventing posterior collapse.

This repository provides scripts to:
* train Calibrated Robust CVAE with $\sigma$-calibration or $\beta$ annealing.
* simulate, load, and preprocess datasets used in the experiments section.
* generate graphs shown in the experiments section.

In addition, this repo includes
* Data: `Data.py` lists contains all the dataset used in the experiments
* Model: `model_xxx.py` lists contains all variant of CVAE.

For more details, see our full paper [here](https://openreview.net/forum?id=VIkycTWDWo&noteId=NPhSjt6Cq8).

<!-- Setup -->
## Getting Started
### Prerequisites 
The following module is required on your machine:
* Linux (Ubuntu or CenOS)/Windows (10 or 11)
* Python 3.7
* Packages in 37Evn.yml

### Installation
Download the repo manually (as a .zip file) or clone it using Git.
```sh
git clone https://github.com/ChuanhuiLiu/calibrated_cvae.git
```
We recommend to use the following command to create a conda environment in anaconda prompt: 
```sh
conda create -n 37Env python=3.7
conda env update -n 37Env -f 37Env.yml 
```
We use the GPU version of Pytorch 1.8.2 because the CUDA 11.1 is pre-installed and can't be changed in cluster GPUs. 

* Please change the following command based on your specification to install Pytorch 
```sh
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```
Don't forget activate the environment and update ipykernel for Juypter Notebook:
```sh
conda activate 37Env
conda install -n 37Env ipykernel --update-deps --force-reinstall
```
Now select the kernel and run `Env_test.ipynb` in your IDE. If everything looks good, you're all set!

<!-- USAGE EXAMPLES -->
## Quick tutorial for $\sigma$-calibration: 

**In `Demo.ipynb`, we provide a standalone implementation of $\sigma$ calibration, offering a step-by-step guide to calibrate $\sigma$ on the Two-moon dataset using an MLP-based CVAE**

<!--CITATION-->
## Citation:
If you find this project helpful, please consider cite it as:

```
@article{
liu2025doubly,
title={Doubly Robust Conditional VAE via Decoder Calibration: An Implicit KL Annealing Approach},
author={Chuanhui Liu and Xiao Wang},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=VIkycTWDWo},
note={}
}
```
<p align="right">(<a href="#getting-started">Back to Top</a>)</p>