import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('/home/igorbaio/Documentos/DeepLearningA-Z/Base_de_Dados/entradas_breast.csv')
# print(previsores)

classe = pd.read_csv("/home/igorbaio/Documentos/DeepLearningA-Z/Base_de_Dados/saidas_breast.csv")
# print(classe)

def CriarRede():
    classificador = Sequential()
    #camadas ocultas
    #units = (ao numero de colunas + as saidas)/ 2 
    classificador.add(Dense(units = 16, activation = 'relu',
                            kernel_initializer = 'random_uniform', 
                            input_dim = 30 ))
    
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16, activation = 'relu',
                            kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))

    #camada de saida
    classificador.add(Dense(units = 1, activation = 'sigmoid'))

    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)

    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                        metrics = ['binary_accuracy'])

    return classificador

classificador = KerasClassifier(build_fn=CriarRede,epochs = 100,batch_size = 10)

resultados = cross_val_score(estimator = classificador,X=previsores,y= classe,cv = 10,scoring= 'accuracy')

media = resultados.mean()
desvio = resultados.std()