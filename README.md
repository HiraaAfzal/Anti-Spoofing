# Real-time Liveness Detection - Ensuring the Integrity of Facial Recognition Systems
A Course Project as part of course final exam Submitted By:
- Adnan Irshad 
- Hira Afzal

## Abstract
In the face recognition system, one of the common problems that arise are presentation attacks or spoofing attacks, which raises questions about the performance and reliability of face recognition-based systems for ensuring access & security. To tackle this issue, this project proposes an extra security layer of face anti-spoofing in the face recognition process through the implementation of face liveness detection in real time. The proposed system uses two anti-spoofing techniques: Motion analysis by eyes blinks detection & mouth movement/speaking detection and color texture analysis by color models techniques. Motion analysis approach is suitable for detecting only image-based spoofing attacks, but it fails in video-based spoofing attacks. Therefore, along with motion analysis approach, texture analysis approach by using colorspaces technique is used to ensure more robust anti-spoofing protection against mobile image, mobile replay and printed image spoofing attacks. The proposed system uses Eyes/Mouth movements detection along with the YCrCb and CIEluv color spaces to detect face liveness and then combines both histograms of these color spaces into a single feature set, which is used as input to a support vector machine (SVM) classifier for liveness prediction.

<br>
<img align="center" src="diagrams/Color Texture Analysis Diagram.drawio.png" alt="Color Texture Analysis Diagram">
<p align="center">Fig. 1: Color Texture Analysis Diagram</p>
<br>

## Dataset
Since majority databases like CASIA Face Anti-Spoofing and Replay-Attack datasets were private and needed access to train & evaluate classifier, we created our own custom dataset by taking 15 to 20 seconds of videos with different poses and lightening conditions of each case including live face, live replay attack, mobile photo attack, printed image and photo of printed image. Apart from this we also gathered some publicly available datasets as well to prevent bias datasets. After gathering dataset and recording our own videos, we performed preprocessing operations including converting videos into frames, extracting face from frames & cropping faces, and splitting into training and testing datasets. We trained our face liveness classifier on 5957 training images and 1491 testing images. There were a total of three classes including 2555 images of printed attack, 757 images of replay attack and 2645 images of real/live faces.

<br>
<img align="center" src="diagrams/dataset diagram.drawio.png" alt="Dataset Collection">
<p align="center">Fig. 2: Dataset Collection</p>
<br>

## Methodology
The proposed system for liveness detection is implemented in Python3. After gathering & preparing dataset, its development divided into two parts, training liveness detection model and testing model in real time webcam.
In training the liveness classifier, following steps were performed:
1.   Loading the Dataset
2.   Preprocessing: Preprocesses dataset by extracting, cropping, and resizing face into 224 x 224.
3.   Color space conversion: Converted each face image from the RGB color space to both the Y CrCb and CIE Luv* color spaces.
4.   Feature extraction: Calculated six histograms corresponding to each component of the Y CrCb and CIE Luv* color spaces and concatenate them into a feature vector.
5.   Model training: Trained an SVM classifier using the extracted features and labels.
6.   Model evaluation: Evaluate the performance of the model using metrics such as accuracy, precision, recall, and F1-score, confusion matrix and classification report.
7.   Saved the classifier in pickle format.

After training and evaluating classifier for liveness detection, used the trained classifier and tested it in real time via web cam streaming.
1. Captured web cam live streaming using OpenCV.
2. Detect face using Mediapipe library.
3. Check Face Liveness by eye & mouth movement using facial landmarks. 
   1. Get facial landmarks of detected face. 
   2. Get eyes landmarks indexes and calculate the EAR. 
   3. If EAR is less than a certain threshold point count this as eyes blinking.
   4. Get lips landmarks indexes and calculate the MAR. 
   5. If MAR is greater than certain threshold point, count this as mouth open. 
   6. If either of the eyes blinking or mouth movements occurs, then go to next liveness check if no movements occur then mark the current face as spoofed attack in step 5.
4. If liveness occurred in step 4 then check liveness using color texture analysis approach. 
   1. Convert face into Y CrCb. 
   2. Convert face into CIE luv.
   3. Calculate histogram of Y CrCb face
   4. Calculate histogram of CIE luv face
   5. Concatenate calculated histograms & convert into features.
   6. Predict liveness class from trained classifier as Live, Printed Attack or Replay Attack
5. Show the liveness results.


