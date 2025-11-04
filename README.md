# ⚡ energy-simulation 🔋

A Reinforcement Learning project leveraging **Stable-Baselines3** to optimize energy efficiency in realistic **5G network management**.

---

## 🎯 Features

* **Custom 5G Environment (Gymnasium-based)**: Implements a bespoke simulation environment modeling complex 5G network dynamics.
* **🌐 Scenario Support**: Includes predefined, realistic deployment scenarios: **dense urban**, **rural**, and **high-speed railway**.
* **💡 Energy Optimization Focus**: Core objective is minimizing energy consumption while rigorously maintaining Quality of Service (QoS).
* **🤖 Reinforcement Learning Integration**: Supports training and evaluation with state-of-the-art algorithms like **PPO** and **SAC**.
* **📈 Diagnostics and Evaluation Tools**: Comprehensive scripts for analyzing agent performance, debugging, and verifying QoS compliance.

---

## 🏗️ Project Structure
*A high-level overview of the project's file and directory structure.*

| File/Directory | Type | Description |
| :--- | :--- | :--- |
| `fiveg_env.py` | File | **Custom 5G Environment** (Gymnasium-based) implementation. |
| `train.py` | File | Script for **training** new RL agents (PPO, SAC). |
| `evaluate.py` | File | Script for **evaluating** trained models and performance. |
| `diagnose.py` | File | Tools for **debugging** and detailed simulation analysis. |
| `simulation_logic.py` | File | Contains the **core simulation logic** and state transition rules. |
| `scenarios/` | Directory | Stores **predefined 5G deployment scenarios** (JSON configs). |
| `sb3_models/` | Directory | Location for **saved, trained RL models**. |
| `sb3_logs/` | Directory | Location for **TensorBoard logs** and training artifacts. |
| `requirements.txt` | File | Python package dependencies (for `pip install`). |
| `environment.yaml` | File | Conda environment configuration file. |

---

## ⚙️ Installation

### 🧩 Prerequisites
- Python **3.11**
- **Conda** (recommended for environment management)

### 🪜 Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/energy-simulation.git
   cd energy-simulation

2. **Activate the environment**
    ```bash
    conda env create -f environment.yml  
    conda activate energy