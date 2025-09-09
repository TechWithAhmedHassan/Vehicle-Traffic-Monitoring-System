# 🚦 Vehicle Traffic Monitoring & Counting System

This project is a **vehicle traffic monitoring system** built with **Python** and **OpenCV**.  
It detects, tracks, and counts vehicles crossing a virtual line in a video stream.  

The system is useful for **traffic analysis**, **smart city projects**, and **intelligent transportation research**.

---

## ✨ Features

- Vehicle detection using background subtraction  
- Centroid tracking with unique IDs  
- Cross-line counting (avoids duplicate counts)  
- Noise and shadow filtering for accuracy  
- Real-time display with bounding boxes, IDs, and counters  
- Works with any video input (extendable to live camera feeds)  

---

## 📖 How It Works

1. **Background Subtraction** – separates moving vehicles from the static background.  
2. **Contour Extraction** – bounding boxes are drawn around detected vehicles.  
3. **Centroid Tracking** – each vehicle is given a unique ID and tracked.  
4. **Counting Logic** – when a vehicle crosses the counting line, it is added to the counter.  
5. **Output Display** – bounding boxes, IDs, and live counter are shown on the video.  

---

## 📂 Project Overview

- **Input**: Traffic video file (e.g., `video1.mp4`)  
- **Output**: Processed video with bounding boxes, IDs, and a live counter  
- **Display**:  
  - Green boxes highlight vehicles  
  - IDs above each tracked vehicle  
  - Counter at the top showing total number of incoming vehicles  

---

## ⚙️ Requirements

- Python 3.x  
- OpenCV  
- NumPy  

Install dependencies:

```bash
pip install opencv-python numpy
