
# 🐱 Cat Breeds Detection using YOLOv8

This project is a deep learning–based object detection system that identifies and classifies different cat breeds from images using YOLOv8. The model is trained on a custom dataset from Roboflow and can detect multiple cats and predict their breed in real time.

---

## 🚀 Project Overview

The goal of this project is to build an accurate computer vision model that can:

- Detect cats in images
- Classify cat breeds
- Handle multiple cats in one image
- Perform real-time predictions
- Provide bounding box detection with labels

This project demonstrates an end-to-end object detection pipeline including data preparation, model training, validation, and inference.

---

## 🧠 Model Used

- YOLOv8 (You Only Look Once v8)
- Real-time object detection
- High accuracy with fast inference
- Multi-class detection support

---

## 🐾 Cat Breeds Detected

The model detects the following 5 cat breeds:

1. American Curl  
2. Bengal  
3. British Shorthair  
4. Persian  
5. Siamese  

Number of classes (nc): **5**

---

## 📊 Dataset Information

- Source: Roboflow Universe  
- Project Name: cat-breeds-fzcip  
- Workspace: philippine-cat-breeds  
- Version: 3  
- License: CC BY 4.0  

Dataset URL:  
https://universe.roboflow.com/philippine-cat-breeds/cat-breeds-fzcip/dataset/3

---

## 📂 Dataset Structure

```
Cat-Breeds-3/
│
├── train/
│   └── images/
│
├── valid/
│   └── images/
│
└── test/
    └── images/
```

### Paths Used

```
Train: /content/CatBreedsDetection/Cat-Breeds-3/train/images
Validation: /content/CatBreedsDetection/Cat-Breeds-3/valid/images
Test: /content/CatBreedsDetection/Cat-Breeds-3/test/images
```

---

## ⚙️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Google Colab
- Roboflow
- Deep Learning (Computer Vision)

---

## 🏋️ Model Training Pipeline

1. Dataset download from Roboflow  
2. Image annotation and labeling  
3. Train / validation / test split  
4. YOLOv8 model training  
5. Model validation  
6. Performance evaluation  
7. Inference on new images  

---

## 📈 Features

- Multi-class object detection
- Real-time prediction
- Bounding box visualization
- Supports multiple cats in one image
- Easy to deploy
- Custom dataset training

---

## ▶️ How to Run

### Step 1 — Install YOLOv8
```bash
pip install ultralytics
```

### Step 2 — Train Model
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)
```

### Step 3 — Predict
```python
model.predict("image.jpg", show=True)
```

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- Precision
- Recall
- mAP (Mean Average Precision)
- Validation loss

---

## 🎯 Applications

- Pet breed identification systems
- Veterinary assistance tools
- Animal monitoring systems
- Smart camera detection
- Educational AI projects

---

## 📌 Future Improvements

- Add more cat breeds
- Increase dataset size
- Mobile app deployment
- Real-time webcam detection
- Edge device optimization

---

## 👩‍💻 Author

Dipti Verma

---

