# Frequency Domain Topologies
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 3 Hours

## 1. Background / Scenario
Sound is a signal in time, but it's often more informative in the frequency domain. In this exercise, you'll implement the **Short-Time Fourier Transform (STFT)** from scratch on the [ESC-50 (Environmental Sound Classification)](https://www.kaggle.com/datasets/mmoreaux/esc50-environmental-sound-classification) dataset.

## 2. Problem Statement
1.  **The Fourier Integral**: Implement a basic Discrete Fourier Transform (DFT).
2.  **Windowing**: Use a Hamming or Hanning window to prevent spectral leakage.
3.  **Spectrogram Generation**: Convert a 1D audio wave into a 2D intensity map (Frequency x Time).

## 3. Dataset Description
**Reference:** [ESC-50 Dataset](https://www.kaggle.com/datasets/mmoreaux/esc50-environmental-sound-classification)
- **Features:** 50 classes of 5-second audio recordings.

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: DFT Matrix (30 pts):** Implement $X_k = \sum x_n \exp(-i 2\pi k n / N)$.
*   **Task 4.2: Sliding Window STFT (40 pts):** Segment the audio and stack DFT outputs.
*   **Task 4.3: Log-Mel Scaling (20 pts):** Convert frequencies to the "Mel" scale which mimics human hearing.
*   **Task 4.4: Audio Visualization (10 pts):** Plot the waveform vs the generated Spectrogram.

## 5. Constraints & Technical Rules
- **No Librosa:** For the DFT core, you must use `NumPy` math, not pre-built signal processing libraries.
- **Language:** Python.
