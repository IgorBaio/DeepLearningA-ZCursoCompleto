import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('/home/igorbaio/Documentos/DeepLearningA-Z/Base_de_Dados/autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis =1)
base = base.drop('dateCreated', axis =1)
base = base.drop('nrOfPictures', axis =1)
base = base.drop('postalCode', axis =1)
base = base.drop('lastSeen', axis =1)
#print(base['name'].value_counts())
base = base.drop('name', axis =1)

#print(base['seller'].value_counts())
base = base.drop('seller', axis =1)

#print(base['offerType'].value_counts())
base = base.drop('offerType', axis =1)

#Pre-processamento de valores inconsistentes
i1 = base.loc[base.price <= 10]
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base[base.price < 350000]

#Pre-processamento de valores faltantes
print(base.loc[pd.isnull(base['vehicleType'])])
print(base['vehicleType'].value_counts()) #limousine
print(base.loc[pd.isnull(base['gearbox'])])
print(base['gearbox'].value_counts()) #manuell
print(base.loc[pd.isnull(base['model'])])
print(base['model'].value_counts()) #golf
print(base.loc[pd.isnull(base['fuelType'])])
print(base['fuelType'].value_counts()) #benzin
print(base.loc[pd.isnull(base['notRepairedDamage'])])
print(base['notRepairedDamage'].value_counts()) #nein


valores = {'vehicleType': 'limousine',
            'gearbox': 'manuell',
            'model': 'golf',
            'fuelType': 'benzin',
            'notRepairedDamage': 'nein'
            }

base = base.fillna(value = valores)

#Pre processamento label encoder
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

label_encoder_previsores = LabelEncoder()
previsores[:, 0] = label_encoder_previsores.fit_transform(previsores[:,0])
#previsores[0]
previsores[:, 1] = label_encoder_previsores.fit_transform(previsores[:,1])
previsores[:, 3] = label_encoder_previsores.fit_transform(previsores[:,3])
previsores[:, 5] = label_encoder_previsores.fit_transform(previsores[:,5])
previsores[:, 8] = label_encoder_previsores.fit_transform(previsores[:,8])
previsores[:, 9] = label_encoder_previsores.fit_transform(previsores[:,9])
previsores[:, 10] = label_encoder_previsores.fit_transform(previsores[:,10])

#Pre processamento one hot encoder
#previsores[0:20]
# 0 == 0 0 0
# 2 == 0 1 0
oneHotEncoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0,1,3,5,8,9,10])],remainder='passthrough') #OneHotEncoder(categories = [0, 1, 3, 5, 8, 9, 10])
previsores = oneHotEncoder.fit_transform(previsores).toarray()

#Rede neural
regressor = Sequential()
#316(entradas) + 1(saida -> preco) /2 = 158.5
regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
regressor.add(Dense(units = 158, activation = 'relu'))
regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', 
                  metrics = ['mean_absolute_error'])

regressor.fit(previsores, preco_real, batch_size =300, epochs = 100)

previsoes = regressor.predict(previsores)





















