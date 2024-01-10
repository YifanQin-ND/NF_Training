# Negative Feedback Training: A Novel Concept to Improve Robustness of NVCIM DNN Accelerators
[[Arxiv](https://arxiv.org/abs/2305.14561)] [[Notre Dame SCL Lab](https://www3.nd.edu/~scl/)]

# Overview
Compute-in-memory (CIM) accelerators built upon non-volatile memory (NVM) devices excel in energy efficiency and latency when performing Deep Neural Network (DNN) inference, thanks to their in-situ data processing capability. However, the stochastic nature and intrinsic variations of NVM devices often result in performance degradation in DNN inference. Introducing these non-ideal device behaviors during DNN training enhances robustness (**noise-injection training**), but has drawbacks. Here, we draw inspiration from the control theory and propose a novel training concept: **Negative Feedback Training (NFT)** leveraging the multi-scale noisy information captured from network. We develop two specific NFT instances, Oriented Variational Forward (OVF) and Intermediate Representation Snapshot (IRS).

<p align="center">
  <img src="https://github.com/YifanQin-ND/NF_Training/files/13882423/two_instances.pdf" alt="Two Instances" width="50%" height="50%">
</p>

## 1. Oriented Variational Forward (OVF)

OVF generates feedback using the less presentative outputs from oriented variational forwards, which are forward processes with device variations larger than inference variations. Using them as negative feedback, OVF inhibits the backbone’s deviation from the optimal optimization direction.

## 2. Intermediate Representation Snapshot (IRS)

IRS provides the means to observe and regulate the data representations within a neural network during training. In this way, internal perturbations can be reflected in the objective and mitigated through negative feedback.


# Robustness Improvements

Our negative feedback training improves the accuracy of inference w/ noise and decreases the uncertainty (expected KL divergence).

![2in1_results.pdf](https://github.com/YifanQin-ND/NF_Training/files/13882469/2in1_results.pdf)

![EKL.pdf](https://github.com/YifanQin-ND/NF_Training/files/13882481/EKL.pdf)
