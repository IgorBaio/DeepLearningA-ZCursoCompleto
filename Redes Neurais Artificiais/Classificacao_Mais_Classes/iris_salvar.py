import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

base = pd.read_csv('/home/igorbaio/Documentos/DeepLearningA-Z/Base_de_Dados/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()

classificador.add(Dense(units = 8, activation = 'relu', input_dim = 4 ))
classificador.add(Dropout(0.25))
classificador.add(Dense(units = 8, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation = 'softmax'))

classificador.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

classificador.fit(previsores, classe, batch_size = 10, epochs = 1000)

classificador_json = classificador.to_json()
with open("classificador_breast_iris.json", "w") as json_file:
    json_file.write(classificador_json)

classificador.save_weights('classificador_breast_iris.h5')