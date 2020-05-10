import numpy as np
from keras.models import model_from_json
import pandas as pd

arquivo = open('classificador_breast_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast_iris.h5')

novo = np.array([[6.2, 2.1, 1, 4]])

previsao = classificador.predict(novo)

previsao = previsao > 0.5

if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')
    

novo1 = np.array([[3.2, 4.5, 0.9, 1.1]])
previsao1 = classificador.predict(novo1)
previsao1 = (previsao1 > 0.5)
if previsao1[0][0] == True and previsao1[0][1] == False and previsao1[0][2] == False:
    print('Iris setosa')
elif previsao1[0][0] == False and previsao1[0][1] == True and previsao1[0][2] == False:
    print('Iris virginica')
elif previsao1[0][0] == False and previsao1[0][1] == False and previsao1[0][2] == True:
    print('Iris versicolor')