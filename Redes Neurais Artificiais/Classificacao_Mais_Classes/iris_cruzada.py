import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('/home/igorbaio/Documentos/DeepLearningA-Z/Base_de_Dados/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelEnconder = LabelEncoder()
classe = labelEnconder.fit_transform(classe)

def CriarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation = "relu", input_dim = 4))    
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 4, activation = "relu"))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation = "softmax"))
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5 )
    classificador.compile(optimizer = otimizador, loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

    return classificador

classificador = KerasClassifier(build_fn= CriarRede, epochs = 1000, batch_size = 10 )

resultados = cross_val_score(estimator = classificador, X=previsores, y=classe, cv = 10, scoring='accuracy')

media = resultados.mean()

desvio = resultados.std()
