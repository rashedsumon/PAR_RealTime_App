# Passenger Attribute Recognition (PAR) - Real-Time System

## Overview
This project implements a real-time computer vision system that classifies visual attributes of passengers detected in video streams. It uses cropped person images and predicts attributes like:

- Gender
- Age range
- Hair length
- Upper/Lower clothing
- Accessories (hats, glasses, bags, etc.)

## Features
- Real-time inference on live or recorded video
- Multi-task model with multi-head classification
- Lightweight backbone for speed: MobileNetV3 / EfficientNet-Lite / ResNet18
- CPU/GPU optimization: Quantization / TensorRT / Async inference
- Easy deployment with Streamlit

## Folder Structure
