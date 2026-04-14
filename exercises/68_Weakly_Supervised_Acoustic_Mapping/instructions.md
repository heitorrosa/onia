# Weakly Supervised Acoustic Mapping
**Difficulty Level:** Phase 3 (Advanced) | **Time Limit:** 3 Hours

## 1. Background / Scenario
Collecting fine-grained labels (start/end times) for audio is expensive. **MIL (Multiple Instance Learning)** allows training with "bag" labels (e.g., "This 5-minute file contains a bird"). You will use the [BirdCLEF Audio Dataset](https://www.kaggle.com/c/birdclef-2021) to implement weakly supervised classification.

## 2. Problem Statement
1.  **Instance Splitting**: Divide a long audio clip into 10 segments (instances).
2.  **Pooling Mechanisms**: Implement **Max Pooling** and **Attention Pooling** to aggregate segment predictions into a single file-level label.
3.  **Top-k Logic**: Only train on the top $k$ most "confident" segments in a clip.

## 3. Dataset Description
**Reference:** [BirdCLEF](https://www.kaggle.com/c/birdclef-2021).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Segment Dataloader (20 pts):** Crop audio into 5-second overlapping instances.
*   **Task 4.2: Global Pooling Head (40 pts):** Custom `nn.Module` that learns which segment is the "most important."
*   **Task 4.3: Binary Cross-Entropy with MIL (30 pts):** Apply loss at the bag level.
*   **Task 4.4: Localization Inference (10 pts):** Identify which specific 5-second block contains the bird call.

## 5. Constraints & Technical Rules
- **Model:** Pre-trained EffNet or MobileNet is allowed for features.
- **Framework:** `PyTorch`.
