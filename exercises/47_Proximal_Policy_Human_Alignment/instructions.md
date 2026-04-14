# Proximal Policy Human Alignment
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
RLHF (Reinforcement Learning from Human Feedback) uses **PPO (Proximal Policy Optimization)** to ensure AI models behave safely. In this exercise, you'll implement a simplified PPO agent to solve the [CartPole Balancing](https://www.kaggle.com/datasets/pankajmandale/cartpole-dataset) problem.

## 2. Problem Statement
1.  **Policy & Value Networks**: Build an Actor (Action distribution) and a Critic (Value estimation).
2.  **Clipped Loss**: Implement the PPO objective function to prevent the policy from shifting too radically in one update.
3.  **Advantage Estimation**: Use GAE (Generalized Advantage Estimation) to calculate the "Advantage" of actions.

## 3. Dataset Description
**Reference:** OpenAI Gym CartPole-v1.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Actor-Critic Architecture (30 pts):** Dual-head PyTorch model.
*   **Task 4.2: Ratio Clipping Logic (40 pts):** Implement $L^{CLIP} = E[\min(r_t \hat{A}_t, clip(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)]$.
*   **Task 4.3: Advantage Calculation (20 pts):** Implement the GAE recursive sum.
*   **Task 4.4: Training Curve (10 pts):** Achieve a score of 500 consistently.

## 5. Constraints & Technical Rules
- **Stability:** Show how the clipping parameter $\epsilon$ prevents policy collapse.
- **Framework:** `PyTorch` and `Gymnasium`.
