# GravityAI: Physics-Constrained Deep Learning Framework for Inverse Gravimetry

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official implementation of the paper: **"GravityAI: A Physics-Constrained Deep Learning Framework for Non-Contact Volumetric Mass Estimation via Inverse Gravimetry"** (Submitted to *Computers & Geosciences*, Elsevier).

##  Overview

Inverse gravimetry—the recovery of a 3D mass distribution from surface gravitational measurements—is a severely ill-posed problem. **GravityAI** is a novel physics-constrained deep learning framework that resolves the full 3-D volumetric mass distribution from sparse, noisy gravitational measurements **without requiring labeled density data** at any stage of training. 

Instead of minimizing a supervised regression loss, GravityAI is trained exclusively by embedding **Newton’s law of universal gravitation** directly into the mean-squared-error objective. This restricts the learnable hypothesis space to physically admissible solutions, achieving superior generalization compared to purely data-driven models.

## 🚀 Key Features

- **Label-Free Training:** Eliminates the need for labeled density training data.
- **Physics-Constrained Loss:** Embeds Newton's law directly into the optimization loop.
- **GPU-Accelerated Engine:** Uses a pre-computed distance tensor for vectorized forward model evaluation.
- **Robust Performance:** Achieves **95.73% ± 1.93%** mass accuracy with 36 sensors under 5% Gaussian noise, outperforming classical Tikhonov regularization.
- **Generalization:** Avoids the memorization pitfalls of supervised networks (which collapse to ~63% on unseen structures).

## 🛠️ Prerequisites

To run this code, you need **Python 3.8+** and the following libraries. You can install them using the provided requirements file:

```bash
pip install -r requirements.txt
Core Dependencies:
torch (PyTorch)
numpy
matplotlib
scikit-image (for SSIM calculation)
📂 Repository Structure
text

12345
How to Run
1. Quick Test (Recommended for First-Time Users)
To verify that the framework is working correctly on your machine without waiting for the full training, run the lightweight quick-test script. This runs a minimal configuration (4x4x4 grid) and completes in a few seconds:
bash

1
2. Main Experiment (Full Paper Reproduction)
To run the full experiment as described in the paper (16x16x16 grid, 36 sensors, 5000 epochs), execute the main script:
bash

1
Note: The full experiment requires a GPU for optimal performance. The script will automatically search for the best random seed before starting the final training loop.
📈 Expected Results
When running the main experiment with 36 sensors and 5% AWGN, you should expect results similar to the following:
Metric
GravityAI (Ours)
Tikhonov Baseline
Mass Accuracy
95.73% ± 1.93%
93.02% ± 4.67%
SSIM
0.203 ± 0.063
0.071 ± 0.038
📝 Citation
If you use this code or framework in your research, please consider citing our paper:
bibtex

1234567
👨‍💻 Author & Contact
Jabril Ramdan
Department of Computer Science, Faculty of Science and Natural Resources
Aljafara University, Alzahraa, Tripoli, Libya
Email: jabril@aju.edu.ly
ORCID: 0000-0001-7011-8694
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
🙏 Acknowledgments
The author acknowledges the support of Aljafara University, Tripoli, Libya.
