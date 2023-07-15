from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import cv2
from sklearn.model_selection import train_test_split


trainDf = pd.read_excel("C:/Users/lucas/OneDrive/Documentos/TG/IA/TG/Dados/DadosParaUso/Treino.xlsx")
testeDf = pd.read_excel("C:/Users/lucas/OneDrive/Documentos/TG/IA/TG/Dados/DadosParaUso/Teste.xlsx")


imagens = []
rotulos = []

# Tamanho desejado para as imagens (por exemplo, 100x100)
novo_tamanho = (230, 230)

# Loop pelas linhas do DataFrame
for index, row in trainDf.iterrows():
    nome_imagem = row['Imagem']
    print(nome_imagem)
    caminho_imagem = 'C:/Users/lucas/OneDrive/Documentos/TG/Imagens_GMAW/Experimentos/' + nome_imagem  # Atualize o caminho para o diretório das imagens

    # Carregar a imagem usando o OpenCV
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    # Redimensionar a imagem
    imagem_redimensionada = cv2.resize(imagem, novo_tamanho)

    # Converter a imagem para um array NumPy
    array_imagem = np.array(imagem_redimensionada)

    # Adicionar o array da imagem e o rótulo às listas
    imagens.append(array_imagem)
    rotulos.append(row['Imagem'])


X_train = np.array(imagens)
X_train = X_train.reshape(X_train.shape[0], 230, 230, 1)
X_train = X_train / 255.0
X_train = X_train.astype('float32')

y_train = trainDf[["percentageLeftPoca","percentageWidthPoca","percentageLeftArame","percentageWidthArame","percentageFinalArame"]].values
y_train = np.array(y_train)
y_train = y_train.astype('float32')


imagens_teste = []
rotulos_teste = []
for index, row in testeDf.iterrows():
    nome_imagem = row['Imagem']
    caminho_imagem = 'C:/Users/lucas/OneDrive/Documentos/TG/Imagens_GMAW/Experimentos/' + nome_imagem  
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    imagem_redimensionada = cv2.resize(imagem, novo_tamanho)
    array_imagem = np.array(imagem_redimensionada)
    imagens_teste.append(array_imagem)
    rotulos_teste.append(row['Imagem'])

X_teste = np.array(imagens_teste)
X_teste = X_teste.reshape(X_teste.shape[0], 230, 230, 1)
X_teste = X_teste / 255.0
#X_teste = X_teste.astype('float32')

y_teste = testeDf[["percentageLeftPoca","percentageWidthPoca","percentageLeftArame","percentageWidthArame","percentageFinalArame"]].values
y_teste = np.array(y_teste)
#y_teste = y_teste.astype('float32')



imagens_treino, imagens_validacao, y_treino, y_validacao = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

sera = y_treino[:, 0]

inputs = Input(shape=(230,230,1))
regressor = Conv2D(16, (3, 3), activation='relu')(inputs)
regressor = MaxPooling2D((2, 2))(regressor)
regressor = Conv2D(32, (3, 3), activation='relu')(regressor)
regressor = MaxPooling2D((2, 2))(regressor)
regressor = Conv2D(32, (3, 3), activation='relu')(regressor)
regressor = MaxPooling2D((2, 2))(regressor)
regressor = Conv2D(64, (3, 3), activation='relu')(regressor)
regressor = MaxPooling2D((2, 2))(regressor)
regressor = Flatten()(regressor)
regressor = Dense(units = 30, activation='tanh', kernel_initializer='truncated_normal')(regressor)
regressor = Dropout(0.2)(regressor)
regressor = Dense(units = 30, activation='tanh', kernel_initializer='truncated_normal')(regressor)
regressor = Dropout(0.2)(regressor)
regressor = Dense(units = 30, activation='tanh', kernel_initializer='truncated_normal')(regressor)
regressor = Dropout(0.2)(regressor)
regressor = Dense(units = 30, activation='tanh', kernel_initializer='truncated_normal')(regressor)
regressor = Dropout(0.2)(regressor)
output1 = Dense(1, activation='sigmoid')(regressor)
output2 = Dense(1, activation='linear')(regressor)
output3 = Dense(1, activation='linear')(regressor)
output4 = Dense(1, activation='linear')(regressor)
output5 = Dense(1, activation='linear')(regressor)



model = Model(inputs=inputs, outputs=[output1,output2,output3,output4,output5])


model.compile(optimizer='SGD', loss='mean_squared_error')

history = model.fit(
    imagens_treino,[y_treino[:, 0], y_treino[:, 1], y_treino[:, 2],y_treino[:, 3], y_treino[:, 4]],
    epochs=1,
    batch_size=1,
    validation_data=(imagens_validacao,[y_validacao[:,0],y_validacao[:,1],y_validacao[:,2],y_validacao[:,3],y_validacao[:,4]])
)

y_pred = model.predict(X_teste)

leftpoca_pred, widthpoca_pred, leftarame_pred, widtharame_pred, alturaarame_pred = y_pred
plt.plot(leftpoca_pred, label='saida rede')
plt.plot(y_teste[:,1],label="saida real")
plt.xlabel('dados')
plt.ylabel('saida')
plt.legend()
plt.show()