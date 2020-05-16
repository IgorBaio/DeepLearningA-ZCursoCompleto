import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('/home/igorbaio/Documentos/DeepLearningA-Z/Base_de_Dados/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelEnconder = LabelEncoder()
classe = labelEnconder.fit_transform(classe)


def CriarRede(optimizer,activation ,neurons1,neurons2 ):
    classificador = Sequential()
    classificador.add(Dense(units = neurons1, activation = activation, input_dim = 4))
    classificador.add(Dropout(0.25))
    classificador.add(Dense(units= neurons2, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation='softmax'))
    
    classificador.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classificador


classificador = KerasClassifier(build_fn = CriarRede)

parametros = {'batch_size': [10,30],
              'epochs': [1000],
              'optimizer': ['adam'],
              'activation': ['relu', 'sigmoid', 'tanh'],
              'neurons1': [4,8],
              'neurons2':[4,8]
              }

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           cv = 5
                           )

grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_

melhor_precisao = grid_search.best_score_