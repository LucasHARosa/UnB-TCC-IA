"""
Esse código serve para converter o arquivo .json com as saidas dos rotulos para um arquivo .csv
"""

import json
import csv

# Nome do arquivo .json com as saidas dos rotulos
with open('export-2023-05-15T12_11_38.510Z.json') as json_file: 
    data = json.load(json_file) 
  
data_file = open('data_file.csv', 'w') 
csv_writer = csv.writer(data_file) 

# Cada objeto se tornará uma linha no arquivo .csv
for emp in data: 
    csv_writer.writerow(emp.values()) 
  
data_file.close() 