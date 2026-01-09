# Real Time Object Detection Using OpenCV

## Mini Project – VTU (21CSL65)  
**Computer Graphics & Image Processing Laboratory**

---

## 📌 Project Overview

This project implements a **Real-Time Object Detection System** using **OpenCV** and a pre-trained **MobileNet SSD (Single Shot MultiBox Detector)** deep learning model.  
The system captures live video from a webcam, detects objects in real time, and displays bounding boxes along with class labels and confidence scores.

The project demonstrates how computer vision and deep learning techniques can be combined to build efficient and accurate object detection systems suitable for real-time applications.

---

## 🎯 Objectives

- To perform **real-time object detection** using a webcam
- To identify and classify objects using **MobileNet SSD**
- To display bounding boxes, labels, and confidence scores
- To maintain real-time performance with minimal latency
- To understand practical applications of **Computer Vision & Image Processing**

---

## 🧠 Technologies Used

- **Python 3**
- **OpenCV (cv2)**
- **MobileNet SSD**
- **NumPy**
- **Imutils**
- **Caffe Deep Learning Framework**

---

## 🏗️ Project Structure



Real-Time-Object-Detection/
│
├── README.md
│
├── report/
│ ├── Mini_Project_Report.pdf
│
├── src/
│ ├── object_detection.py
│ └── requirements.txt
│
├── models/
│ ├── MobileNetSSD_deploy.prototxt
│ └── MobileNetSSD_deploy.caffemodel
│
├── screenshots/
│ ├── output_detection.png
│ ├── known_face.png
│ └── unknown_face.png
│
└── .gitignore


---

## ⚙️ System Requirements

### Hardware
- Intel Core i3 or higher
- Minimum 4 GB RAM
- Webcam (720p or above)

### Software
- Windows 10 / Linux
- Python 3.6+
- OpenCV 4.5+
- NumPy
- Imutils

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install -r requirements.txt

2️⃣ Run the Object Detection Script
python object_detection.py

3️⃣ Exit the Application

Press q to stop the video stream.
