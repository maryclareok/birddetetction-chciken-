# Chicken Disease Classification & Stolen Chicken/Intruder Detection

## Overview

This repository **combines two functionalities**:
1. **Detecting sick or healthy chickens** using a **VGG16-based transfer learning** approach on an image dataset.
2. **Monitoring** a video feed for **stolen chickens** and **intruders** using **MobileNetSSD** object detection.

## Dataset for Chicken Classification

- **Broiler-Chicken-Healthy-and-Sick**  
  [Download here](https://universe.roboflow.com/technicalresearch/broiler-chicken-healthy-and-sick/dataset/1/download)

## Part 1: Chicken Disease Classification

### 1. CSV Structure: `filename`, `Healthy`, `Sick`  
- We **convert** `(Healthy=1,Sick=0)` → `"Healthy"`, `(Healthy=0,Sick=1)` → `"Sick"` in a **single** `"label"` column.  
- We **exclude** ambiguous rows `(0,0)` or `(1,1)` if they’re invalid.

### 2. Building the VGG16 Model  
- **Transfer Learning**: Load **VGG16** (include_top=False), then add **GlobalAveragePooling**, **Dense**, **Dropout** layers.  
- **Initial Training**: Freeze base VGG16 layers, train with `lr=1e-4`.  
- **Fine-Tuning**: Unfreeze last block, train with `lr=1e-5`.

### 3. Final Metrics & Plots  
- Evaluated on a 20% validation split, achieving ~**97% accuracy** and **F1**.  
- Plots show **training vs. validation accuracy & loss** during both initial training and fine-tuning.  
- Saves the final model as `chicken_disease_model.h5`.

## Part 2: Stolen Chicken & Intruder Detection

### 1. MobileNetSSD for Object Detection  
- **OpenCV** (`cv2.dnn`) loads a **pretrained MobileNetSSD** (`MobileNetSSD_deploy.caffemodel`).  
- Class IDs: `bird=3` → “chicken”, `dog=12`, `person=15`, `cat=8` → “intruders.”

### 2. Video Processing  
- Iterates each frame:
  1. Resizes and blobs the image, forward it through MobileNetSSD.  
  2. Classifies bounding boxes with confidence > 0.6.  
  3. Tracks chickens (dictionary of unique IDs) and intruders.  
  4. Alerts if a chicken from the previous frame is missing → possible theft.  
  5. Alerts if a new intruder is detected.

### 3. Data Visualization  
- **pandas** DataFrames store detection events.  
- Plots:
  - **Number of detections** over time.  
  - **Confidence** over time.  
  - **Missing chicken** & **intruder** alerts.  
  - **Heatmap** of detection locations in the frame.

## How to Use

1. **Install Dependencies**  
   ```bash
   pip install tensorflow pandas opencv-python-headless numpy seaborn matplotlib scikit-learn
   ```
2. **Update Paths**  
   - For classification, ensure `base_dir` and `_classes.csv` point to your dataset.  
   - For intruder detection, set `video_path` to your local video file.
3. **Run Chicken Classification**  
   - Train the VGG16 model, verify logs for accuracy/f1, and check final metrics.
4. **Run Stolen Chicken / Intruder Script**  
   - Processes each frame of the specified video, outputs alerts on missing chickens or new intruders.
5. **Visualize Plots**  
   - Model training curves (accuracy, loss).  
   - Final single-point metrics (accuracy, precision, recall, F1).  
   - Intruder/chicken detection timeline, confidence, alerts, and heatmap.

## Results

- **Chicken Classification**: ~97% Accuracy, F1, Recall, and Precision on the validation set.  
- **Intruder Detection**: Real-time bounding box draws, console alerts, plus data-driven visual plots showing detection events over time.
