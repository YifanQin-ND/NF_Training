# Negative Feedback Training: A Novel Concept to Improve Robustness of NVCIM DNN Accelerators
[[Arxiv](https://arxiv.org/abs/2305.14561)] [[Notre Dame SCL Lab](https://www3.nd.edu/~scl/)] [![LICENSE](https://img.shields.io/github/license/JoeyBling/hexo-theme-yilia-plus "LICENSE")](./LICENSE "LICENSE") 


# Overview
Compute-in-memory (CIM) accelerators built upon non-volatile memory (NVM) devices excel in energy efficiency and latency when performing Deep Neural Network (DNN) inference, thanks to their in-situ data processing capability. However, the stochastic nature and intrinsic variations of NVM devices often result in performance degradation in DNN inference. Introducing these non-ideal device behaviors during DNN training enhances robustness (**noise-injection training**), but has drawbacks. Here, we draw inspiration from the control theory and propose a novel training concept: **Negative Feedback Training (NFT)** leveraging the multi-scale noisy information captured from network. We develop two specific NFT instances, Oriented Variational Forward (OVF) and Intermediate Representation Snapshot (IRS).

<div algin="center">
<img src="figures/overview.svg" width="400"> <img src="figures/methods.svg" width="360">
</div>

## 1. Oriented Variational Forward (OVF)

OVF generates feedback using the less presentative outputs from oriented variational forwards, which are forward processes with device variations larger than inference variations. Using them as negative feedback, OVF inhibits the backbone’s deviation from the optimal optimization direction.

## 2. Intermediate Representation Snapshot (IRS)

IRS provides the means to observe and regulate the data representations within a neural network during training. In this way, internal perturbations can be reflected in the objective and mitigated through negative feedback.


# Robustness Improvements

Our negative feedback training improves the accuracy of inference w/ noise and decreases the uncertainty (expected KL divergence).

<p align="center">
  <img src="figures/results.svg" alt="results" width="50%">
</p>

<p align="center">
  <img src="figures/EKL.png" alt="EKL" width="40%">
</p>


# Usage
We provide one example to illustrate the usage of the code.
For the IRS instance, we run resent8 with device relative variation 0.3. Training for 200 epochs and then conduct 200 times inference w/ noise Monte Carlo simulation (parameters can be modified in config).
```
python res18_main.py \
--mode tnt \
--type irs \
--dataset mnist \
--var1 0.3 \
--var2 0.3 \
--beta 1e-1 \
--num 1 \
--mark 1.1
```
--num and --mark are artificial labelings, with no practical significance for experiments. For more details of args, check the codes.

# Reference
If you find NFT useful or relevant to your research, please kindly cite our paper:
```
@article{qin2023negative,
  title={Negative Feedback Training: A Novel Concept to Improve Robustness of NVCiM DNN Accelerators},
  author={Qin, Yifan and Yan, Zheyu and Wen, Wujie and Hu, Xiaobo Sharon and Shi, Yiyu},
  journal={arXiv preprint arXiv:2305.14561},
  year={2023}
}
```

# License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
