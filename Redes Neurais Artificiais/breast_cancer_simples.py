import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('/home/igorbaio/Documentos/DeepLearningA-Z/Base_de_Dados/entradas_breast.csv')
# print(previsores)

classe = pd.read_csv("/home/igorbaio/Documentos/DeepLearningA-Z/Base_de_Dados/saidas_breast.csv")
# print(classe)
previsores_treinamento, previsores_test, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()
#camadas ocultas
#units = (ao numero de colunas + as saidas)/ 2 
classificador.add(Dense(units = 16, activation = 'relu',
                        kernel_initializer = 'random_uniform', 
                        input_dim = 30 ))

classificador.add(Dense(units = 16, activation = 'relu',
                        kernel_initializer = 'random_uniform'))


#camada de saida
classificador.add(Dense(units = 1, activation = 'sigmoid'))

otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)

classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

#classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
#                      metrics = ['binary_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 100)

pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
pesos1 = classificador.layers[1].get_weights()
print(pesos1)
print(len(pesos1))
pesos2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(previsores_test)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
resultado = classificador.evaluate(previsores_test, classe_teste)
print("Precisao:",precisao)
print("Matriz de confusao:")
print(matriz)
print("Resultado de taxa de Erro:",resultado)