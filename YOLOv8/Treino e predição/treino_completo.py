# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1GOfPHY6ilzsTrs54dJ2glNf1SbB1D3ws
"""


from google.colab import drive
drive.mount('/content/drive')

!nvidia-smi

!pip install ultralytics

from ultralytics import YOLO

"""Couston Training Module"""
# Modelo treinado com 5000 imagens

!yolo task=detect mode=train model=yolov8n.pt data=/content/drive/MyDrive/datasets/Yolo/data.yaml epochs=50 imgsz=256