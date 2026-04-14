# Early Stopping MLP Integration
**Difficulty Level:** Phase 2 (Deep Learning) | **Time Limit:** 1 Hour

## 1. Background / Scenario
Training for too many epochs leads to overfitting. Early Stopping monitors a validation metric (usually <code>val_loss</code>) and halts training if it doesn't improve for a sequence of epochs (patience). This ensures your model generalizes well.

## 2. Problem Statement
Integrate an Early Stopping callback mechanism into a standard Training/Validation loop for an MLP. You'll use the [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset and save the version of the model that achieved the minimum validation loss.

## 3. Dataset Description
Use the [Health Risk](https://www.kaggle.com/datasets/vuppalaadithyasairam/heart-disease-dataset) (13 features, 1025 items) dataset. Divide into Train (80%) and Test (20%).

## 4. Subtasks & Point Distribution
* **Task 4.1: Validation Monitor (30 pts):** Each epoch, compute <code>val_loss</code> on the 20% test set by putting the model in <code>eval()</code> mode and using <code>torch.no_grad()</code>.
* **Task 4.2: Patience Counter (40 pts):** Create a <code>best_loss</code> variable. If <code>val_loss < best_loss</code>, reset a <code>patience_counter</code> to 0 and save <code>model.state_dict()</code>. If not, increment the counter.
* **Task 4.3: Training Halt (30 pts):** Break the training loop immediately when <code>patience_counter >= 10</code>.

## 5. Constraints & Technical Rules
* Strictly forbidden: Use of <code>PyTorch Lightning</code> or <code>Scikit-Learn</code> EarlyStopping. All logic must be raw PyTorch in standard <code>for-epochs</code> loop style.

## 6. Evaluation Criteria
* Training should stop around Epoch 40-70 if configured correctly.
* Reach 0.84 Accuracy on the "Best Saved Model" after training terminates.

## 7. Deliverables
* <code>early_stop_mlp.py</code>: The PyTorch script with patience logic.
* <code>learning_curves.png</code>: Plot showing Training vs Validation Loss over epochs until the stop occurs.
