# Recurrent State Memory Matrices
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 3 Hours

## 1. Background / Scenario
RNNs (Recurrent Neural Networks) struggle with "Long-term Dependencies." LSTMs solve this using "Forget Gates." In this exercise, you'll use the [Daily Climate Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) to forecast temperature.

## 2. Problem Statement
1.  **Recurrent Pipeline**: Implement an `LSTM` or `GRU` model.
2.  **Sliding Window Generation**: Transform the time-series into blocks of (Past 7 Days -> Present Day).
3.  **Stateful Training**: Learn how to pass the `hidden_state` between batches correctly.

## 3. Dataset Description
**Reference:** [Daily Climate in Delhi](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)
- **Features:** Humidity, Meantemp, Windspeed.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Windowed Dataloader (30 pts):** Custom logic to create overlapping sequences.
*   **Task 4.2: LSTM Model (40 pts):** Multiple layers with a final regressive linear head.
*   **Task 4.3: Many-to-One Prediction (20 pts):** Predict only the next day.
*   **Task 4.4: Future Horizon Walk (10 pts):** Use your model recursively to predict the next 3 days.

## 5. Constraints & Technical Rules
- **Normalization:** Time-series are highly sensitive; use `MinMaxScaler`.
- **Framework:** `PyTorch`.
