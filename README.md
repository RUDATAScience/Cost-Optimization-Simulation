# Advanced Stochastic Deterioration Prediction & PINN-Based Infrastructure Asset Management

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository provides a comprehensive suite of simulation and machine learning models aimed at resolving the critical global challenge of aging social infrastructure. By bridging **Civil Engineering** (infrastructure maintenance and public health risk assessment) and **Applied Mathematics** (computational science and stochastic processes), this project addresses the fundamental limitations of traditional asset management.

Real-world infrastructure deterioration inherently involves **extreme risks** (sudden collapses or jump phenomena) that cannot be accurately predicted using sparse, analog inspection records. Mathematically, these sudden deteriorations are modeled using Partial Integro-Differential Equations (PIDEs) with state-dependent Poisson jumps. Due to the "non-locality" of the integral terms in PIDEs, traditional analytical approximations (like perturbation expansion) completely break down. 

To overcome these barriers, this repository implements large-scale Monte Carlo simulations for non-linear Stochastic Differential Equations (SDEs) and utilizes **Physics-Informed Neural Networks (PINNs)** to solve inverse problems, ensuring absolute safety and cost optimization.

## Repository Structure & Modules

The repository consists of four primary analytical scripts, each representing a distinct phase of the research:

### 1. `main1.py` - Continuous Degradation & FPT Simulation
This script models the continuous daily degradation of infrastructure before reaching a critical tipping point (State C).
* **Core Logic:** Simulates a non-linear SDE using the Euler-Maruyama method to track the degradation state over time[cite: 3].
* **Key Features:** Calculates the First Passage Time (FPT) to the threshold State C[cite: 3]. It mathematically proves the limitations of traditional methods by comparing Monte Carlo simulation distributions directly against analytical solutions derived from perturbation expansion[cite: 3].

### 2. `main2.py` - Jump-Diffusion Grid Search (Extreme Value Collapse)
This script focuses on the catastrophic collapse phase occurring after the infrastructure surpasses the tipping point (State C to State D).
* **Core Logic:** Models a Jump-Diffusion SDE over a 72-hour disaster scenario, incorporating a Poisson process to simulate sudden structural failures[cite: 4].
* **Key Features:** Executes a massive grid search over multiple combinations of environmental noise ($\sigma$) and jump intensity ($\lambda$)[cite: 4]. It generates scatter plots demonstrating "negative resilience" and histograms revealing extreme value fat-tail risks[cite: 4].

### 3. `main3.py` - Integrated Phase 1 & 2 Simulation
This module serves as an integrated pipeline that consolidates the continuous degradation model and the post-tipping point jump model.
* **Core Logic:** Runs Phase 1 (continuous SDE) and Phase 2 (Grid Search) sequentially within a unified computational environment[cite: 5].
* **Key Features:** Automatically generates and exports comprehensive datasets (CSVs) and comparative visualizations, packaging them into a downloadable ZIP archive for seamless analysis[cite: 5].

### 4. `main4.py` - PINN-Based Inverse Problem Analysis
This script utilizes advanced AI architectures to reconstruct hidden physical parameters from highly scarce observation data, simulating real-world constraints (e.g., analog paper records).
* **Core Logic:** Implements a Physics-Informed Neural Network (PINN) using PyTorch[cite: 6]. 
* **Key Features:** Solves the inverse problem by estimating the true physical parameters (degradation rate $k$ and environmental noise $\sigma$) from an artificially sparsified dataset of only 100 observation points[cite: 6]. 
* **Loss Function:** The network is trained using a composite loss function combining **Data Loss** (MSE against sparse observations) and **PDE Loss** (residuals of the Fokker-Planck governing equation calculated via automatic differentiation)[cite: 6].

## Scientific Contributions

1. **Breakdown of Analytical Derivations:** The simulations in `main1.py` and `main3.py` demonstrate that while perturbation expansion works for standard Gaussian degradation, it fails for non-local jump phenomena[cite: 3, 5]. 
2. **Quantification of Extreme Risks:** `main2.py` proves that deferring maintenance beyond the tipping point exposes infrastructure to catastrophic, fat-tail societal losses[cite: 4].
3. **Overcoming Data Scarcity:** `main4.py` proves that PINNs can accurately reconstruct underlying physical parameters by embedding governing equations directly into the neural network, bypassing the need for massive, high-frequency digital datasets[cite: 6].

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy pandas matplotlib scipy torch

```

Note: All scripts are designed to be fully compatible with Google Colab. They will automatically detect the Colab environment and prompt a direct download of the generated results (ZIP files).

## Usage

Execute the scripts individually via the command line. Each script will automatically generate visualizations (PNGs), raw data (CSVs), and package them into a ZIP archive.

```bash
python main1.py
python main2.py
python main3.py
python main4.py

```

## License

This project is licensed under the MIT License.
