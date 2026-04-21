# 🐍 Snake RL Comparison Lab

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rl-project-nihar-walawalkar.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A professional, high-performance Reinforcement Learning laboratory designed to compare Tabular and Deep RL algorithms in the classic Snake environment. This project features a completely modernized, glassmorphic UI with real-time analytics and simultaneous multi-agent gameplay.

🚀 **Live Demo:** [rl-project-nihar-walawalkar.streamlit.app](https://rl-project-nihar-walawalkar.streamlit.app/)

---

## 🧬 Supported Algorithms

Explore how different agents learn to survive and hunt:

-   **Q-Learning (Off-Policy):** The classic temporal difference algorithm using the Bellman equation to find the optimal action-selection policy.
-   **SARSA (On-Policy):** A more conservative learner that updates its Q-values based on the action it *actually* takes, making it safer near walls during exploration.
-   **Double Q-Learning:** Solves the classic overestimation bias by using two decoupled Q-tables—one for action selection and another for evaluation.
-   **DQN (Deep Q-Network):** Implements a Neural Network as a function approximator, featuring **Experience Replay** and **Frozen Target Networks** for stable, deep reinforcement learning.

---

## ✨ Key Features

-   **Professional Dark UI:** A premium, state-of-the-art interface built with vanilla CSS overrides for Streamlit.
-   **Simultaneous Live Gameplay:** Watch all trained agents play side-by-side in real-time with adjustable frame rates.
-   **Detailed Performance Analytics:** 
    -   Smoothed Learning Curves (Reward, Score, Steps).
    -   Q-Value Distributions (Visualizing overestimation bias).
    -   Convergence Analysis & Summary Metrics.
-   **Persistence:** Save trained models as `.pkl` (Tabular) or `.pt` (DQN) files and reload them later to skip retraining.
-   **Optimized Training:** Multi-step training frequency and memory-efficient replay buffers for fast convergence.

---

## 🛠️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Nihar-Walawalkar/RL-Mini-Project.git
cd snake-rl
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Linux/macOS
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 🖥️ Project Structure

-   `app.py`: The main Streamlit dashboard and UI logic.
-   `agents.py`: Implementations of Tabular RL agents (Q-Learning, SARSA, Double Q).
-   `dqn_agent.py`: PyTorch implementation of the Deep Q-Network.
-   `snake_env.py`: Custom logic for the Snake environment tailored for RL states.
-   `style.css`: Custom CSS for the advanced dark theme.

---

## 🤝 Contributing
Feel free to open issues or submit pull requests to improve the agents, environment, or UI!

---
*Created as part of an RL Mini Project.*
