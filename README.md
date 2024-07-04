### Overview

This code combines two functionalities: detecting sick or healthy chickens using image classification and monitoring for stolen chickens and intruders using object detection in a video feed.

### Explanation

#### Detection of Sick or Healthy Chickens

##### Importing Libraries

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
```

- **TensorFlow and Keras**: Deep learning frameworks for model development and training.
- **ImageDataGenerator**: Tool for data augmentation and preprocessing of images.
- **MobileNetV2**: Pre-trained CNN model used as a base for transfer learning.
- **Matplotlib**: Library for plotting graphs and visualizations.
- **OS**: Module for interacting with the operating system, used for file operations.

##### Loading and Preprocessing Data

```python
# Loading dataset
train_data_dir = 'dataset/train'
test_data_dir = 'dataset/test'
```

- Specifies directories containing training and testing data.

```python
# Preprocessing data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
```

- Configures data generators for training and testing with image augmentation and preprocessing.

##### Building and Training the Model

```python
# Building the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
```

- Loads MobileNetV2 as the base model without the top layer for transfer learning.

```python
# Adding custom layers
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation='sigmoid')(head_model)

model = Model(inputs=base_model.input, outputs=head_model)
```

- Extends the base model with custom layers for fine-tuning.

```python
# Freezing base layers
for layer in base_model.layers:
    layer.trainable = False
```

- Freezes the weights of the pre-trained layers to prevent retraining.

```python
# Compiling the model
opt = Adam(lr=1e-4)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
```

- Compiles the model with Adam optimizer and binary cross-entropy loss for binary classification.

```python
# Training the model
history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=len(test_data),
    epochs=20,
    verbose=1
)
```

- Trains the model on training data for 20 epochs with validation on test data.

```python
# Plotting accuracy and loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

- Plots training and validation accuracy and loss over epochs to evaluate model performance.

#### Stolen Chicken and Intruder Detection

##### Importing Libraries

```python
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

- **OpenCV (cv2)**: Library for computer vision tasks, including object detection.
- **Time**: Provides time-related functions.
- **NumPy**: Essential for numerical operations and array handling.
- **Matplotlib**: Used for plotting graphs and visualizations.
- **Pandas**: Library for data manipulation and analysis.

##### Load the Pre-trained MobileNet SSD Model

```python
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
```

- Loads a pre-trained MobileNet SSD model for object detection.

##### Define Class Labels and IDs

```python
class_labels = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
                6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable',
                12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
                17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
```

- Maps numeric class IDs to human-readable labels for the MobileNet SSD model.

```python
CHICKEN_CLASS_ID = 3  # 'bird'
INTRUDER_CLASS_IDS = [12, 15, 8]  # 'dog', 'person', 'cat'
```

- Defines class IDs for chickens and intruders based on the MobileNet SSD model.

##### Video Source and Initialization

```python
video_path = '0704.mp4'  # Update with your video path
cap = cv2.VideoCapture(video_path)
```

- Specifies the path to the video file and initializes video capture.

##### Initialize Tracking and Plotting Data

```python
chicken_dict = {}
intruder_dict = {}

detection_data = []
missing_chickens_data = []
intruder_alerts_data = []
detection_positions = []
```

- Sets up dictionaries and lists to track chickens, intruders, and data for plotting.

##### Process Each Frame

```python
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
```

- Begins reading frames from the video.
- `ret` indicates if the frame was successfully read.
- `frame` holds the current frame from the video.

##### Prepare Frame for Detection

```python
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)
```

- Resizes the frame to 300x300 pixels for efficient processing.
- Converts the frame into a blob format suitable for input to the neural network.

##### Perform Object Detection

```python
    net.setInput(blob)
    detections = net.forward()
```

- Sets the input to the neural network with the prepared blob.
- `detections` contains the detected objects in the frame.

##### Process Detections

```python
    current_chickens = {}
    current_intruders = {}

    timestamp = time.time() - start_time
```

- Initializes dictionaries to track chickens and intruders detected in the current frame.
- Calculates the timestamp for the current frame.

##### Loop Over Detections

```python
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x, y, w, h = box.astype(int)
```

- Iterates through all detected objects.
- Extracts confidence levels and bounding box coordinates if confidence is above 60%.

##### Identify and Track Chickens and Intruders

```python
            unique_id = f"{x}_{y}"

            if class_id == CHICKEN_CLASS_ID:
                current_chickens[unique_id] = (x, timestamp)
                color = (0, 255, 0)  # Green for chickens
                label = 'Chicken'
            elif class_id in INTRUDER_CLASS_IDS:
                current_intruders[unique_id] = (x, timestamp)
                color = (0, 0, 255)  # Red for intruders
                label = class_labels[class_id]
            else:
                continue
```

- Generates unique IDs based on object positions to track objects.
- Identifies and tracks chickens and intruders based on their class IDs.
- Draws bounding boxes around detected objects with colors indicating chickens (green) and intruders (red).

##### Draw Bounding Boxes

```python
            cv2.rectangle(frame, (x, y), (w, h), color, 2

)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detection_data.append({'time': timestamp, 'type': label, 'confidence': confidence})
            detection_positions.append((x, y))
```

- Draws rectangles and adds labels with object type and confidence levels.
- Records detection data for visualization.

##### Alert for Missing Chickens

```python
    for chicken_id in list(chicken_dict.keys()):
        if chicken_id not in current_chickens:
            missing_chickens_data.append({'time': timestamp, 'chicken_id': chicken_id})
            print(f"Alert: Chicken {chicken_id} is missing!")
            del chicken_dict[chicken_id]
```

- Checks if previously detected chickens are missing in the current frame.
- Logs and prints an alert if a chicken is missing.

##### Alert for New Intruders

```python
    for intruder_id in current_intruders:
        if intruder_id not in intruder_dict:
            intruder_alerts_data.append({'time': timestamp, 'intruder_id': intruder_id})
            print(f"Alert: Intruder {intruder_id} detected!")
```

- Checks if new intruders appear in the current frame.
- Logs and prints an alert if a new intruder is detected.

##### Update Tracking Dictionaries

```python
    chicken_dict.update(current_chickens)
    intruder_dict.update(current_intruders)
```

- Updates dictionaries with current frame's detected chickens and intruders.

##### Display Frame and Check for Exit

```python
    cv2.imshow('Chicken and Intruder Detection', frame)
    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break
```

- Displays the processed frame in a window.
- Press 'Esc' to exit the loop.

##### Release Resources

```python
cap.release()
cv2.destroyAllWindows()
```

- Releases video capture and closes OpenCV windows.

##### Convert Data to DataFrames

```python
detection_df = pd.DataFrame(detection_data)
missing_chickens_df = pd.DataFrame(missing_chickens_data)
intruder_alerts_df = pd.DataFrame(intruder_alerts_data)
```

- Converts recorded data into DataFrames for easier plotting.

##### Plot the Data

```python
plt.figure(figsize=(18, 10))

# 1. Number of Detections Over Time
plt.subplot(2, 3, 1)
detection_count = detection_df.groupby(['time', 'type']).size().unstack(fill_value=0)
detection_count.plot(ax=plt.gca(), title='Number of Detections Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Count')

# 2. Confidence Levels Over Time
plt.subplot(2, 3, 2)
for label, group in detection_df.groupby('type'):
    plt.plot(group['time'], group['confidence'], label=label)
plt.title('Confidence Levels Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Confidence')
plt.legend()

# 3. Missing Chickens Alerts
plt.subplot(2, 3, 3)
if not missing_chickens_df.empty:
    missing_chickens_df['time'] = pd.to_datetime(missing_chickens_df['time'], unit='s')
    missing_count = missing_chickens_df.groupby(missing_chickens_df['time']).size()
    missing_count.plot(ax=plt.gca(), title='Missing Chickens Alerts')
plt.xlabel('Time')
plt.ylabel('Count')

# Add more plots for intruder alerts and other relevant data

plt.tight_layout()
plt.show()
```

- Creates a figure for plotting multiple graphs.
- Plots data on detections, confidence levels, and alerts for missing chickens over time.
