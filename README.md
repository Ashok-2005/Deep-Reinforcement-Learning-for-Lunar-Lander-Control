# 🚀 STM-DQN: Deep Reinforcement Learning for Lunar Lander Control

A Deep Reinforcement Learning project that compares **Q-Learning**, **Deep Q-Network (DQN)**, and a proposed **Short-Term Memory Deep Q-Network (STM-DQN)** for autonomous lunar landing in the OpenAI Gym LunarLander-v3 environment.

The proposed STM-DQN enhances the traditional DQN by incorporating a Short-Term Memory (STM) module to capture temporal dependencies, leading to improved landing stability, higher cumulative rewards, and better convergence.

---

## 📌 Overview

Safe spacecraft landing is one of the most challenging problems in autonomous control due to complex dynamics, delayed rewards, and uncertain environments.

This project investigates three reinforcement learning approaches:

- Q-Learning
- Deep Q-Network (DQN)
- Short-Term Memory Deep Q-Network (STM-DQN)

The models were trained and evaluated in the **OpenAI Gym LunarLander-v3** environment to compare their learning performance, reward optimization, and landing stability.

---

# 🎯 Objectives

- Implement Q-Learning for Lunar Lander control.
- Develop a Deep Q-Network (DQN) agent.
- Propose an STM-DQN architecture with temporal memory.
- Compare all models under identical training conditions.
- Improve landing accuracy and training stability.
- Analyze reward convergence and agent performance.

---

# ✨ Features

- Lunar Lander autonomous control
- Q-Learning implementation
- Deep Q-Network (DQN)
- Proposed STM-DQN architecture
- Experience Replay
- Target Network
- ε-Greedy Exploration
- Short-Term Memory for temporal learning
- Reward visualization
- Performance comparison of all models

---

# 🛠️ Technologies Used

- Python
- PyTorch
- OpenAI Gym (Gymnasium)
- NumPy
- Matplotlib
- TQDM
- Google Colab

---

# 🌍 Environment

**OpenAI Gym – LunarLander-v3**

### Observation Space

- 8 Continuous Features
  - X Position
  - Y Position
  - X Velocity
  - Y Velocity
  - Angle
  - Angular Velocity
  - Left Leg Contact
  - Right Leg Contact

### Action Space

| Action | Description |
|---------|-------------|
| 0 | Do Nothing |
| 1 | Fire Left Orientation Engine |
| 2 | Fire Main Engine |
| 3 | Fire Right Orientation Engine |

---

# 🧠 Models Implemented

## 1️⃣ Q-Learning

Traditional table-based Reinforcement Learning algorithm.

### Characteristics

- Q-Table
- Bellman Equation
- ε-Greedy Exploration
- State Discretization

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Episodes | 20,000 |
| Learning Rate | 0.1 |
| Gamma | 0.99 |
| Epsilon | 1.0 → 0.05 |

---

## 2️⃣ Deep Q-Network (DQN)

Neural-network-based Q-Learning capable of handling continuous state spaces.

### Components

- Q-Network
- Target Network
- Replay Buffer
- Mini-Batch Learning
- ε-Greedy Policy

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Steps | 200,000 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Replay Buffer | 100,000 |
| Gamma | 0.99 |
| Target Update | Every 1000 Steps |

---

## 3️⃣ Proposed STM-DQN

The proposed STM-DQN extends DQN by introducing a **Short-Term Memory (STM)** layer to retain temporal information from recent states.

Unlike DQN, which treats every state independently, STM-DQN learns sequential dependencies, enabling more stable and context-aware decision making.

### Architecture

```
State (8 Features)
        │
        ▼
 Short-Term Memory Layer
 (LSTM / GRU)
        │
        ▼
 Fully Connected Layers
        │
        ▼
     Q Values
        │
        ▼
 Action Selection
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Episodes | 2,500 |
| Learning Rate | 1e-4 |
| Gamma | 0.99 |
| Batch Size | 64 |
| Replay Buffer | 100,000 |
| STM Hidden Units | 128 |
| Target Update | Every 1000 Steps |

---

# ⚙️ Training Pipeline

```
Environment
      │
      ▼
Current State
      │
      ▼
Agent
(Q-Learning / DQN / STM-DQN)
      │
      ▼
Select Action
      │
      ▼
Environment
      │
      ▼
Reward + Next State
      │
      ▼
Replay Buffer
      │
      ▼
Model Update
```

---

# 📊 Results

## Experimental Comparison

| Model | Mean Reward | Standard Deviation |
|--------|------------:|-------------------:|
| Q-Learning | -227.77 | 141.69 |
| DQN | -24.23 | 48.34 |
| STM-DQN | **181.80** | **113.68** |

---

## 📈 Performance Summary

| Model | Performance |
|--------|-------------|
| Q-Learning | Poor convergence |
| DQN | Better stability |
| STM-DQN | Best reward and stable learning |

---

# 🏆 Key Findings

- STM-DQN achieved the highest average reward.
- Temporal memory significantly improved decision making.
- DQN outperformed Q-Learning in continuous environments.
- Replay Buffer and Target Network stabilized training.
- STM-DQN produced smoother and more reliable landings.

---

# 📈 Evaluation Metrics

The models were evaluated using:

- Mean Episode Reward
- Standard Deviation
- Learning Stability
- Landing Success Rate
- Reward Convergence

---

# 🔮 Future Work

- Double DQN (DDQN)
- Dueling DQN
- Prioritized Experience Replay
- PPO and SAC algorithms
- Transformer-based Reinforcement Learning
- Real-world robotics applications
- Autonomous drone navigation
- Planetary landing missions

---

# 👨‍💻 Authors

- **Venkata Ashok Adithya**
- **Dinesh Reddy**
- **Dhanush Kumar Reddy**
- **Kaseeswar Reddy**


**School of Computer Science and Engineering**

**VIT-AP University**

---

# 📚 References

- OpenAI Gym / Gymnasium Documentation
- Sutton & Barto – Reinforcement Learning: An Introduction
- Mnih et al. – Human-Level Control through Deep Reinforcement Learning
- LunarLander-v3 Environment Documentation
