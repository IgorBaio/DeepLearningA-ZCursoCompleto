#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor


# In[6]:


base = pd.read_csv('C:\\Users\\Igori\\Documents\\Projects\\DeepLearningA-ZCursoCompleto\\Base_de_Dados\\autos.csv', encoding = 'ISO-8859-1')
print(base)


# In[7]:


base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)


# In[8]:


print(base['name'].value_counts())


# In[9]:


base = base.drop('name', axis = 1)


# In[10]:


print(base['seller'].value_counts())


# In[11]:


base = base.drop('seller', axis = 1)


# In[12]:


print(base['offerType'].value_counts())


# In[13]:


base = base.drop('offerType', axis = 1)


# In[14]:


print(base.columns)
print(len(base))


# In[15]:


#Pre-processamento de valores inconsistentes
i1 = base.loc[base.price <= 10]
print(i1)


# In[16]:


base = base[base.price > 10]
print(len(base))


# In[17]:


i2 = base.loc[base.price > 350000]
print(i2)


# In[18]:


base = base.loc[base.price < 350000]
print(len(base))


# In[19]:


#Pre-processamento de valores faltantes
print(base.loc[pd.isnull(base['vehicleType'])])
print(base['vehicleType'].value_counts()) #limousine


# In[21]:


print(base.loc[pd.isnull(base['gearbox'])])


# In[23]:


print(base['gearbox'].value_counts())#manuell


# In[24]:


print(base.loc[pd.isnull(base['model'])])


# In[25]:


print(base['model'].value_counts())#golf


# In[27]:


print(base.loc[pd.isnull(base['fuelType'])])


# In[28]:


print(base['fuelType'].value_counts())#benzin


# In[30]:


print(base.loc[pd.isnull(base['notRepairedDamage'])])


# In[31]:


print(base['notRepairedDamage'].value_counts())#nein


# In[32]:


valores = {
    'vehicleType': 'limousine',
    'gearbox': 'manuell',
    'model': 'golf',
    'fuelType':'benzin',
    'notRepairedDamage':'nein'
    
}


# In[33]:


base = base.fillna(value = valores)


# In[37]:


#Pre processamento label encoder
previsores = base.iloc[:,1:13].values
preco_real = base.iloc[:, 0].values
print(previsores[0])


# In[39]:


label_encoder_previsores = LabelEncoder()
previsores[:, 0] = label_encoder_previsores.fit_transform(previsores[:,0])
previsores[:,1] = label_encoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = label_encoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = label_encoder_previsores.fit_transform(previsores[:,5])
previsores[:,8] = label_encoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = label_encoder_previsores.fit_transform(previsores[:,9])
previsores[:,10] = label_encoder_previsores.fit_transform(previsores[:,10])


# In[40]:


#Pre processamento one hot encoder
previsores[0:20]
#0 == 0 0 0
#2 == 0 1 0


# In[43]:


oneHotEncoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories = 'auto'), [0,1,3,5,8,9,10])], remainder = 'passthrough')


# In[44]:


previsores = oneHotEncoder.fit_transform(previsores).toarray()


# In[56]:


print(previsores.shape[1])


# In[57]:


def CriarRede(drop1,drop2, loss):
    regressor = Sequential()
    #(316 + 1 ) /2 == 158.5 -> 159
    regressor.add(Dense(units = 159, activation = 'relu', input_dim = 316))
    regressor.add(Dropout(drop1))
    regressor.add(Dense(units = 159, activation = 'relu'))
    regressor.add(Dropout(drop2))
    regressor.add(Dense(units = 1, activation = 'linear'))
    
    regressor.compile(loss= loss, optimizer = 'adam', metrics = ['mean_absolute_error'])
    
    return regressor


# In[58]:


regressor = KerasRegressor(build_fn = CriarRede)


# In[59]:


parametros = {
    'batch_size':[300],
    'epochs':[100],
    'loss':['mean_squared_error', 'mean_absolute_error','mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge' ],
    'drop1': [0.2,0.3],
    'drop2':[0.2,0.3]
}


# In[60]:


grid_search = GridSearchCV( estimator = regressor,
                          param_grid = parametros,
                           cv = 5
                          )


# In[61]:


grid_search = grid_search.fit(previsores, preco_real)


# In[62]:


melhores_parametros = grid_search.best_params_
print(melhores_parametros)#{'batch_size': 300, 'drop1': 0.2, 'drop2': 0.2, 'epochs': 100, 'loss': 'squared_hinge'}


# In[65]:


melhor_precisao = grid_search.best_score_
print(melhor_precisao)#0.0


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




