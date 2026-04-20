# PawsitiveHealth
AI Vet: Pawsitive Health 🐾
Real-time Pet Health Monitoring using Computer Vision & Deep Learning
📌 Project Overview
AI Vet is an intelligent diagnostic tool designed to provide early-stage health triage for pets and stray dogs. By utilizing Computer Vision and Machine Learning, the app analyzes posture and physical metrics to detect potential health issues before they become emergencies.
This project was inspired by my dog, Bira, and aims to bridge the gap in emergency veterinary care by connecting pet owners with verified paravet contacts and pet blood banks.
🚀 Key Features
• Postural Analysis: Uses OpenCV to calculate Edge Density and posture metrics to identify signs of physical discomfort or lethargy.
• Real-time Inference: Powered by YOLOv8-Nano for high-speed object detection on edge devices.
• Intelligent Triage: Employs a Random Forest Classifier to analyze extracted features and categorize health status with high reliability.
• Emergency Resource Hub: A built-in directory of verified paravet contacts and pet blood banks for immediate crisis intervention.
🛠️ Tech Stack
• Language: Python
• Vision Engine: OpenCV, YOLOv8-Nano
• Machine Learning: Supervised Learning (Random Forest)
• Data Handling: Pandas, NumPy
• Environment: VS Code, Git/GitHub
🧩 Technical Logic
The core "Diagnostic Engine" works on a multi-stage pipeline:
1.	Detection: Identifying the subject using the YOLOv8-Nano model.
2.	Preprocessing: Applying Gaussian blurring and grayscale conversion to reduce noise from "unclean" real-world backgrounds.
3.	Feature Extraction: Calculating Edge Density (ratio of white pixels to total pixels) and stochastic posture scores.
4.	Classification: A Random Forest model processes these features to handle variability in lighting and movement, preventing overfitting and providing stable health predictions.
