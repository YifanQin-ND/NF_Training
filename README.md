# Negative Feedback Training: A Novel Concept to Improve Robustness of NVCIM DNN Accelerators
[[Arxiv](https://arxiv.org/abs/2305.14561)] [[Notre Dame SCL Lab](https://www3.nd.edu/~scl/)]


# Overview
Compute-in-memory (CIM) accelerators built upon non-volatile memory (NVM) devices excel in energy efficiency and latency when performing Deep Neural Network (DNN) inference, thanks to their in-situ data processing capability. However, the stochastic nature and intrinsic variations of NVM devices often result in performance degradation in DNN inference. Introducing these non-ideal device behaviors during DNN training enhances robustness (**noise-injection training**), but has drawbacks. Here, we draw inspiration from the control theory and propose a novel training concept: **Negative Feedback Training (NFT)** leveraging the multi-scale noisy information captured from network. We develop two specific NFT instances, Oriented Variational Forward (OVF) and Intermediate Representation Snapshot (IRS).

<table>
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/YifanQin-ND/NF_Training/files/13882561/overview.pdf" alt="overview" width="80%" height="80%">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/YifanQin-ND/NF_Training/files/13882423/two_instances.pdf" alt="Two Instances" width="90%" height="90%">
      </p>
    </td>
  </tr>
</table>

## 1. Oriented Variational Forward (OVF)

OVF generates feedback using the less presentative outputs from oriented variational forwards, which are forward processes with device variations larger than inference variations. Using them as negative feedback, OVF inhibits the backbone’s deviation from the optimal optimization direction.

## 2. Intermediate Representation Snapshot (IRS)

IRS provides the means to observe and regulate the data representations within a neural network during training. In this way, internal perturbations can be reflected in the objective and mitigated through negative feedback.


# Robustness Improvements

Our negative feedback training improves the accuracy of inference w/ noise and decreases the uncertainty (expected KL divergence).

<table>
  <tr>
    <td>
      <p align="center">
        <img src="https://github.com/YifanQin-ND/NF_Training/files/13882469/2in1_results.pdf" alt="2in1_results" width="80%" height="80%">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://github.com/YifanQin-ND/NF_Training/files/13882481/EKL.pdf" alt="EKL" width="80%" height="80%">
      </p>
    </td>
  </tr>
</table>

# Usage
We provide one example to illustrate the usage of the code.
For the IRS instance, we run vgg8 with device relative variation 0.3. Training for 200 epochs and then conduct 200 times inference w/ noise Monte Carlo simulation (parameters can be modified in config).
```
python vgg8_main.py \
--mode tnt \
--type irs \
--dataset mnist \
--var1 0.3 \
--var2 0.3 \
--beta 1e-2 \
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
