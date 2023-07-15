"""
Esse código serve para transformar um arquivo csv com as saidas dos rotulos para o formato do YOLOv8

"""
import pandas as pd

planilha = pd.read_excel('teste.xlsx')
print(planilha)

def convert_to_yolov8_format(row):
  classe1 = 0
  classe2 = 1

  # Calcule as coordenadas normalizadas para a primeira classe
  x1_center = row['yolo_xPocaCenter']
  y1_center = row['yolo_yPocaCenter']
  width1 = row['yolo_xPocaw']
  height1 = row['yolo_yPocah']

  # Calcule as coordenadas normalizadas para a segunda classe
  x2_center = row['yolo_xArameCenter']
  y2_center = row['yolo_yArameCenter']
  width2 = row['yolo_xAramew']
  height2 = row['yolo_yArameh']

  # Retorne as coordenadas no formato do YOLOv8
  return f"{classe1} {x1_center} {y1_center} {width1} {height1}\n{classe2} {x2_center} {y2_center} {width2} {height2}"



for index, row in planilha.iterrows():
  image_path = row['Imagem']
  image_name = image_path.split('/')[-1].split('.')[0]
  
  

  yolov8_labels = convert_to_yolov8_format(row)

  with open(f'{image_name}.txt', 'w') as file:
      file.write(yolov8_labels)