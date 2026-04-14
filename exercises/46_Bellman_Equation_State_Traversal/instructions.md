# Bellman Equation State Traversal
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Reinforcement Learning (RL) relies on the **Bellman Equation** to estimate the value of future states. Using the [OpenAI Gym: FrozenLake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) environment, you will implement a Q-Learning agent.

## 2. Problem Statement
1.  **Q-Table Construction**: Build a matrix of `States x Actions`.
2.  **The Bellman Update**: Implement the equation $Q(s,a) = Q(s,a) + \alpha[R + \gamma \max Q(s',a') - Q(s,a)]$.
3.  **Exploration vs Exploitation**: Use an Epsilon-Greedy strategy to balance discovery and reward.

## 3. Dataset Description
**Reference:** FrozenLake Environment.
- **Goal:** Reach the goal without falling into holes.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Q-Table Initialization (20 pts):** Accurate mapping of the 4x4 or 8x8 grid.
*   **Task 4.2: Update Logic (40 pts):** Correct implementation of the temporal difference (TD) error.
*   **Task 4.3: Gamma ($\gamma$) Impact (20 pts):** Compare a "short-sighted" agent ($\gamma=0.1$) vs a "long-sighted" one ($\gamma=0.99$).
*   **Task 4.4: Success Rate Plot (20 pts):** Plot the average reward over 10,000 episodes.

## 5. Constraints & Technical Rules
- **Tooling:** `Gymnasium` (or `Gym`) and `NumPy`.
- **Neural Networks:** For extra credit (not required), implement Deep Q-Learning (DQN).
