from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

trainDf = pd.read_excel("/Treino.xlsx")
testeDf = pd.read_excel("/Teste.xlsx")

imagens = []
tamanho_img = (230, 230)

# Loop pelas linhas do DataFrame
for index, row in trainDf.iterrows():
    nome_imagem = row['Imagem']
    caminho_imagem = '/Experimentos/' + nome_imagem  
    # Carregar a imagem usando o OpenCV
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    # Redimensionar a imagem
    imagem_redimensionada = cv2.resize(imagem, novo_tamanho)
    # Converter a imagem para um array NumPy
    array_imagem = np.array(imagem_redimensionada)
    # Adicionar o array da imagem e o rotulo as listas
    imagens.append(array_imagem)

# Tratamento para o formato dos dados
X_train = np.array(imagens)
X_train = X_train.reshape(X_train.shape[0], 230, 230, 1)
X_train = X_train / 255.0
X_train = X_train.astype('float32')

y_train = trainDf[["percentageLeftArame"]].values
y_train = np.array(y_train)
y_train = y_train 
y_train = y_train.astype('float32')


imagens_teste = []
for index, row in testeDf.iterrows():
    nome_imagem = row['Imagem']
    caminho_imagem = 'Experimentos/' + nome_imagem  
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    imagem_redimensionada = cv2.resize(imagem, novo_tamanho)
    array_imagem = np.array(imagem_redimensionada)
    imagens_teste.append(array_imagem)
    rotulos_teste.append(row['Imagem'])

X_teste = np.array(imagens_teste)
X_teste = X_teste.reshape(X_teste.shape[0], 230, 230, 1)
X_teste = X_teste / 255.0
X_teste = X_teste.astype('float32')

y_teste = testeDf[["percentageLeftArame"]].values
y_teste = np.array(y_teste)
y_teste = y_teste 
y_teste = y_teste.astype('float32')

# Divisão entre treino e validação
imagens_treino, imagens_validacao, y_treino, y_validacao = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Criação do modelo
model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(230, 230, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(units = 30, activation='tanh', kernel_initializer='truncated_normal'))
model.add(Dropout(0.2))
model.add(Dense(units = 30, activation='tanh', kernel_initializer='truncated_normal'))
model.add(Dropout(0.2))
model.add(Dense(units = 30, activation='tanh', kernel_initializer='truncated_normal'))
model.add(Dropout(0.2))
model.add(Dense(units = 30, activation='tanh', kernel_initializer='truncated_normal'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Treinamento
history = model.fit(
    imagens_treino,y_treino,
    epochs=10,
    batch_size=20,
    validation_data=(imagens_validacao,y_validacao)
)

# Avaliação do treinamento
y_pred = model.predict(X_teste)

plt.plot(y_pred, label="saida rede")
plt.plot(y_teste,label="saida real")
plt.xlabel('dados')
plt.ylabel('saida')
plt.legend()
plt.show()

mae = mean_absolute_error(y_teste, y_pred)
mse = mean_squared_error(y_teste, y_pred)
r2 = r2_score(y_teste, y_pred)
print(mae)
print(mse)
print(r2)