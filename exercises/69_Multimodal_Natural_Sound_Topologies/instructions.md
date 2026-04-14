# Multimodal Natural Sound Topologies
**Difficulty Level:** Phase 4 (Extreme) | **Time Limit:** 4 Hours

## 1. Background / Scenario
Humans hear and see at the same time. **Audio-Visual Fusion** models combine these signals. In this challenge, you'll use the [Audio-Visual Event (AVE) Dataset](https://www.kaggle.com/datasets/maricinnamon/audio-visual-event-ave-dataset) to synchronize video frames with audio.

## 2. Problem Statement
1.  **Late Fusion**: Build two encoders (CNN for Video, Spectrogram-CNN for Audio) and concatenate their final vectors.
2.  **Attention-based Fusion**: Use a "Cross-Attention" block where the audio features "look" at the video features to find relevant visual cues.
3.  **Temporal Alignment**: Minimize the distance between audio and video embeddings when they represent the same event.

## 3. Dataset Description
**Reference:** [AVE Dataset](https://www.kaggle.com/datasets/maricinnamon/audio-visual-event-ave-dataset).

## 4. Subtasks & Point Distribution (100 Points)
*   **Task 4.1: Dual-Stream Dataloader (30 pts):** Synchronized reading of .wav and .mp4.
*   **Task 4.2: Cross-Modal Attention (40 pts):** Query = Video, Key/Value = Audio.
*   **Task 4.3: Contrastive Synchronization Loss (20 pts):** Penalize when audio and video drift apart in time.
*   **Task 4.4: Result Analysis (10 pts):** Does adding video improve audio-only accuracy?

## 5. Constraints & Technical Rules
- **Framework:** `PyTorch`.
- **Complexity:** Focus on the "Cross-Attention" implementation math.
