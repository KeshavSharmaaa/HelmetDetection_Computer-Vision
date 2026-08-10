# HelmetDetection_Computer-Vision
# 🪖 SafeRide — Helmet Detection Using Classical Computer Vision

## 📌 About the Project

**SafeRide** is a real-time helmet detection system developed as a **Machine Vision project** to identify whether a two-wheeler rider is wearing a helmet.

The main focus of this project is to perform helmet detection **without using YOLO, deep learning models, or pre-trained object detection models**.

Instead, the system uses **classical Computer Vision and Image Processing techniques** to analyze the visual structure of an image and identify helmet-like regions.

The detection process works by focusing on the upper portion of the image, where a helmet is expected to be located. The system then performs image preprocessing, edge detection, morphological operations, contour analysis, shape filtering, and HSV-based color analysis to determine whether a helmet is present.

The project also includes a **Streamlit-based interface** where the detection system can be tested using:

* 🖼️ Images
* 📹 Recorded videos
* 📡 Live webcam feed

The idea behind the project is to demonstrate that useful real-world vision tasks can be approached using fundamental **Machine Vision techniques**, rather than relying entirely on deep learning.

---

# 🎯 Objectives

The main objectives of SafeRide are:

* Detect helmets using classical computer vision techniques.
* Analyze the shape and appearance of possible helmet regions.
* Process images and video frames automatically.
* Provide real-time helmet detection through a webcam.
* Create a simple interface for testing the detection system.
* Demonstrate practical applications of concepts such as edge detection, morphology, contours, segmentation, and color analysis.

---

# 🧠 How the Detection Works

The detection system follows a sequence of image-processing operations.

```text
Input Image / Video Frame
          ↓
Focus on Upper Region
          ↓
Image Resizing
          ↓
Grayscale Conversion
          ↓
Gaussian Blur
          ↓
Canny Edge Detection
          ↓
Morphological Operations
          ↓
Contour Detection
          ↓
Shape Filtering
          ↓
HSV Color Analysis
          ↓
Helmet Detection
```

The system does not simply look for one specific color or shape.

Instead, several visual characteristics are checked before a region is considered a possible helmet.

---

# 🔍 Machine Vision Techniques Used

## 1. Region of Interest

Since the helmet is expected to be around the rider's head, the system first focuses on the **upper portion of the image** rather than processing the entire frame equally.

This reduces unnecessary processing and allows the detector to concentrate on the region where a helmet is most likely to appear.

---

## 2. Image Resizing

If the input image is relatively small, the upper region is enlarged before processing.

This helps improve the quality of contours and edges that are extracted from the image.

---

## 3. Grayscale Conversion

The selected region is converted from BGR into grayscale.

This makes the image easier to process for edge detection because the algorithm can work with intensity information rather than all three color channels.

---

## 4. Gaussian Blur

A Gaussian blur is applied before edge detection.

This helps reduce small amounts of noise and makes the subsequent edge detection more stable.

---

## 5. Canny Edge Detection

The system uses **Canny edge detection** to identify important boundaries in the image.

These edges are then used to find possible helmet-shaped regions.

---

## 6. Morphological Operations

After obtaining the edges, the system applies morphological operations.

Dilation strengthens the detected edges, while morphological closing helps connect nearby edge regions and fill small gaps.

This makes the resulting contours more suitable for analysis.

---

## 7. Contour Detection

Once the processed edge image is available, the system extracts contours.

Each contour represents a possible object or region that can then be analyzed based on its size and shape.

---

## 8. Area Filtering

Very small contours and extremely large contours are ignored.

The system uses image-size-dependent area limits so that only reasonably sized regions are considered as possible helmet candidates.

---

## 9. Circularity Analysis

One of the important parts of the detector is **contour circularity**.

Circularity is calculated using:

```text
Circularity = 4π × Area / Perimeter²
```

This helps identify contours that have a reasonably rounded or helmet-like shape.

The project uses a relaxed circularity threshold so that angled or oval-shaped helmets can also be considered.

---

## 10. Aspect Ratio Filtering

The bounding box of a candidate region is also checked.

The system expects a possible helmet region to have a reasonably balanced width-to-height ratio, so extremely thin or stretched regions are rejected.

---

## 11. HSV Color Analysis

The candidate region is converted into the **HSV color space**.

Instead of assuming that helmets have one particular color, the system checks for:

* Dark/black regions
* White/light-gray regions
* Strongly saturated colors

This allows the detector to work with helmets of different colors.

---

# 🔄 Fallback Detection

If the main contour-based detection does not find a helmet, the system has a second detection approach.

This fallback method looks for a **large dark rounded region in the upper portion of the frame**.

It uses HSV thresholding followed by morphological operations and contour analysis.

## This is particularly useful for detecting dark or black helmets whose edges may not be very obvious.

# 🖥️ User Interface

The project uses **Streamlit** to provide an interactive interface.

The application is called **SafeRide OS** and provides three main detection modes:

```text
🏠 Home
🖼️ Image Analysis
📹 Video Process
📡 Live Stream
```

The application also provides information about the computer vision techniques being used and the overall system.

---

# 🖼️ Image Analysis

In Image Analysis mode, the user can upload:

* JPG
* JPEG
* PNG

The uploaded image is passed directly to the helmet detection function.

The interface then displays:

```text
Original Image          Detection Result
     ↓                         ↓
   Input                  Helmet / No Helmet
```

If a helmet is detected, the application displays:

**✅ Helmet Detected!**

Otherwise, it reports:

**🚨 Violation: No Helmet Detected!**

---

# 📹 Video Processing

The application can also process recorded videos.

Supported formats include:

* MP4
* AVI
* MOV

The video is read frame by frame using OpenCV, and the helmet detector is applied to each frame.

The processed frames are then displayed directly in the Streamlit interface.

---

# 📡 Live Webcam Detection

The project can also use a webcam for real-time detection.

When **Activate System** is enabled, OpenCV accesses the default camera and continuously processes the incoming frames.

```text
Webcam
   ↓
OpenCV
   ↓
Frame
   ↓
Helmet Detection
   ↓
Processed Frame
   ↓
Streamlit
```

The processed video is displayed live in the application.

---

# 🛠️ Technology Stack

| Technology | Purpose                              |
| ---------- | ------------------------------------ |
| Python     | Main programming language            |
| OpenCV     | Image processing and computer vision |
| NumPy      | Numerical and matrix operations      |
| Streamlit  | Web-based user interface             |
| PIL        | Image loading and handling           |

The project specifically uses **classical computer vision**, rather than YOLO or a trained object-detection model. The core detection logic is implemented directly with OpenCV operations.

---

# 📁 Project Structure

The project can be kept in a simple structure like this:

```text
SafeRide/
│
├── app.py
├── detection.py
├── train_img1.jpg
│
└── README.md
```

### `detection.py`

This contains the actual helmet detection logic.

The main function is:

```python
detect_helmet(frame)
```

It receives an image frame and returns:

```text
Processed frame
+
Detection result
```

The file also contains the fallback detection method and image-processing pipeline.

### `app.py`

This is the Streamlit application.

It handles the interface, image uploads, video processing, and webcam input while calling the detection function from `detection.py`.

---

# 🚀 Setup

Okay, now the easy part.

If you just want to run the project on your laptop, you don't need to install anything crazy.

## 1. Install Python

First, check if Python is already installed:

```bash
python --version
```

If that doesn't work, try:

```bash
py --version
```

If you get a Python version, you're good to go.

---

## 2. Open the Project Folder

Download or clone this project and open the folder in VS Code.

Your folder should contain something like:

```text
app.py
detection.py
```

---

## 3. Install the Libraries

Open the VS Code terminal and run:

```bash
pip install opencv-python numpy streamlit pillow
```

That's pretty much it.

No YOLO model.

No model weights.

No TensorFlow.

No complicated setup.

---

## 4. Run the App

Once the libraries are installed, run:

```bash
streamlit run app.py
```

Streamlit should start the application and give you a local address.

Open that address in your browser.

---

# 🧪 How to Use It

Once the app opens:

### 🏠 Home

You can see the project overview and the technologies used.

### 🖼️ Image Analysis

Upload a rider image and let the system analyze it.

### 📹 Video Process

Upload an MP4, AVI, or MOV video and watch the detection run frame by frame.

### 📡 Live Stream

Turn on **Activate System**, allow camera access if your browser asks, and the system will start processing the webcam feed.

That's it. 😭

---

# ⚠️ Important Note

This project is a **classical computer vision-based detection system**, so its performance depends heavily on the visual conditions of the input.

Things such as:

* Lighting
* Camera angle
* Image quality
* Helmet color
* Background objects
* Rider position
* Helmet visibility

can affect the detection result.

The system is therefore intended primarily as a **Machine Vision project and demonstration of classical image-processing techniques**, rather than a production-ready traffic enforcement system.

---

# 🚀 Future Improvements

The current system can be extended in several ways.

### Better Helmet Detection

More advanced combinations of segmentation, shape analysis, and feature extraction could improve detection in difficult environments.

### Number Plate Detection

A separate classical computer vision pipeline could be developed to locate number plates using:

* Edge detection
* Thresholding
* Contours
* Morphological operations
* Aspect-ratio filtering

### Number Plate Recognition

After detecting the plate, OCR could be added to extract the actual registration number.

### Violation Database

Detected violations could be stored with:

* Date
* Time
* Image
* Detection result

### Improved Live Monitoring

The Streamlit interface could be extended into a complete traffic-monitoring dashboard.

---

# 📚 Machine Vision Concepts Demonstrated

This project brings together several important concepts from the Machine Vision syllabus:

* Image acquisition
* Image preprocessing
* Grayscale conversion
* Gaussian filtering
* Edge detection
* Canny operator
* Thresholding
* HSV color analysis
* Morphological operations
* Contour extraction
* Shape analysis
* Circularity
* Aspect-ratio analysis
* Real-time image processing

---

# ⭐ Conclusion

SafeRide demonstrates how a practical road-safety problem can be approached using **classical Computer Vision techniques**.

Instead of relying on a pre-trained object detection model, the system analyzes image structure directly through edges, contours, morphology, shape characteristics, and color information.

The Streamlit interface makes the system easier to demonstrate by allowing the same detection pipeline to work with **images, recorded videos, and live webcam input**.

The project provides a foundation that can be further expanded into a more complete road-safety monitoring system.
