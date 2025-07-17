# 😷 Face Mask Detection

**Detecting Mask Compliance Using Deep Learning & Computer Vision**




### 🔍 Smart Vision for Safer Spaces

**Face Mask Detection** is a deep learning-based system that automatically detects whether people are wearing face masks in real-time. Trained on labeled datasets with "Mask", "No Mask", and optionally "Improper Mask" classes, this project enables fast, accurate, and practical deployment in surveillance, workplace safety, and public monitoring systems.




## 🚀 Highlights

* 🎯 CNN-based model trained on real-world face mask data
* 👨‍👩‍👧‍👦 Multi-class detection: *Mask*, *No Mask*
* 🖼️ Real-time detection on images 
* 🔍 OpenCV-based face detection preprocessing
* ⚡ Lightweight, fast, and deployment-friendly




## 📁 Dataset Overview

* **Source:** Public face mask detection datasets (e.g. Kaggle)
* **Total Images:** 4,000+ (augmented)
* **Classes:**

  * `With Mask`
  * `Without Mask`
* **Image Size:** 128x128 (resized)
* **Split:** Train / Validation / Test (80/10/10)

> Dataset was cleaned and preprocessed using OpenCV face detection to isolate faces before classification.




## 📊 Model Performance

| Metric    | Value     |
| --------- | --------- |
| Accuracy  | \~97%     |
| Precision | \~96%     |
| Recall    | \~95%     |
| Inference | \~30+ FPS |




## 🧠 How It Works

1. **Face Detection:**
   OpenCV’s Haar Cascade or DNN-based detector is used to isolate faces in frames.

2. **Classification:**
   Detected face regions are passed through a CNN model trained to classify mask status.

3. **Result Rendering:**
   Bounding boxes with labels (`Mask`, `No Mask`) are drawn in real-time on image or webcam frames.

4. **Output:**
   Mask compliance is clearly marked with color-coded boxes and text.




## 📂 Project Structure

```
├── Face_Mask_Detection.ipynb   # Full pipeline: data prep, training, inference
├── model/
│   └── face_mask_model.h5      # Trained Keras model
├── data/
│   ├── with_mask/              # Masked face images
│   └── without_mask/           # Unmasked face images
├── haarcascade_frontalface.xml # Pre-trained face detector
├── detect_mask_video.py        # Real-time video detection script
├── config.yaml                 # Optional config (for expansion)
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```




## 🛠️ Built With

* Python 3.x
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn




## 🧭 Applications

* Workplace mask compliance enforcement
* Retail and public area monitoring
* Smart surveillance systems
* Health monitoring in pandemic scenarios
* Automated access control systems




## 🧩 Future Scope

* [ ] Add class for improper mask usage
* [ ] Export to ONNX or TFLite for mobile deployment
* [ ] Integration with CCTV streams
* [ ] Add audio alert system for violations
