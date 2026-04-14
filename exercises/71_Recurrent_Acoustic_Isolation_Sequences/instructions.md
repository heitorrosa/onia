# Recurrent Acoustic Isolation Sequences
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 5 Hours

## 1. Background / Scenario
"The Cocktail Party Problem": Isolating one primary speaker from a noisy background. Using the [LibriMix Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/librimix-clean-train-100), you will build a **Source Separation** model.

## 2. Problem Statement
1.  **Spectrogram Masking**: Predict a soft mask (0-1) that, when multiplied by the noisy spectrogram, isolates the clean speech.
2.  **Architecture**: Use a Bi-Directional LSTM or a Wave-U-Net.
3.  **SDR (Signal-to-Distortion Ratio)**: Implement the metric used to judge isolation quality.

## 3. Dataset Description
**Reference:** [LibriMix Subset](https://www.kaggle.com/datasets/rashikrahmanpritom/librimix-clean-train-100).
- **Features:** Mixed audio (Speaker A + Speaker B + Noise).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Mixed Audio Data Generator (20 pts):** Functionally sum two clean wav files and add Gaussian noise.
*   **Task 4.2: Masking Network (40 pts):** CNN-LSTM architecture predicting the spectral mask.
*   **Task 4.3: Inverse STFT (20 pts):** Convert the masked spectrogram back into a .wav file.
*   **Task 4.4: Audio Quality Listen-test (20 pts):** Report the SDR improvement before and after isolation.

## 5. Constraints & Technical Rules
- **Frequency:** Use 16kHz sampling rate.
- **Framework:** `PyTorch`.
