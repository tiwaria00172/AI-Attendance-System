# AI Face Recognition Attendance System

An AI-powered attendance system that automatically detects and recognizes multiple faces from classroom or group images and records attendance using a computer-vision inference pipeline.

Built with **Python, OpenCV, YuNet, SFace, ONNX models, NumPy, Pandas, and Streamlit**.

---

## 🚀 Overview

Traditional attendance methods can be time-consuming and prone to manual errors.

This project automates the process by allowing a user to upload a classroom/group image. The system:

1. Detects faces in the image
2. Extracts facial embeddings
3. Compares embeddings with registered students
4. Identifies matching students using cosine similarity
5. Prevents duplicate attendance
6. Records attendance with date and time
7. Displays the processed result through a Streamlit interface

The core focus of the project is the **computer-vision inference pipeline and backend processing logic**, while Streamlit provides a lightweight application interface.

---

## ✨ Features

- 👤 Multiple-face detection
- 🧠 Face recognition using SFace
- 🔍 YuNet lightweight face detection
- 📐 Cosine similarity-based face matching
- ⚙️ Configurable recognition threshold
- 🖼️ Image preprocessing and resizing
- 🛡️ Duplicate attendance prevention
- 📊 Attendance records using CSV
- 👨‍🎓 Student registration system
- 💾 Face embedding storage
- 📅 Date and time-based attendance tracking
- 📱 Streamlit-based interface
- 🔄 Automatic ONNX model download
- 🧩 Modular detection and recognition pipeline

---

## 🏗️ System Architecture

```text
                    Input Image
                         │
                         ▼
              ┌─────────────────────┐
              │  Image Preprocessing │
              │ Resize / Normalize   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   YuNet Detector    │
              │   Face Detection    │
              └──────────┬──────────┘
                         │
                    Detected Faces
                         │
                         ▼
              ┌─────────────────────┐
              │   SFace Recognizer  │
              │ Embedding Extraction│
              └──────────┬──────────┘
                         │
                    Face Embedding
                         │
                         ▼
              ┌─────────────────────┐
              │ Cosine Similarity   │
              │     Matching        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Student Identification│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Duplicate Prevention│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Attendance Record   │
              │ CSV + Date + Time    │
              └─────────────────────┘
