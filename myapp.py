#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st 
import numpy as np
import pandas as pd
import io
import os
import random

# In[5]:


st.title('Menu Semanal según requeirmiento calórico')


# In[ ]:


st.write("""
# El mecanismo Basal
Para hombres:
TMB = (10 x peso en kg) + (6,25 × altura en cm) – (5 × edad en años) + 5

Para mujeres:
TMB = (10 x peso en kg) + (6,25 × altura en cm) – (5 × edad en años) – 161

""")


# In[7]:

peso = st.sidebar.text_area('Ingrese su peso en Kg')
altura = st.sidebar.text_area('Ingrese su altura en cm', height=7)
edad = st.sidebar.text_area('Ingrese su edad en años', height=7)
sexo= st.sidebar.radio('Indique su  sexo', ('Hombre', 'Mujer'))
if sexo == 'Hombre':
    sexo= 'H'
if sexo == 'Mujer':
    sexo = 'M'

def calculo_tmb(peso, altura, edad, sexo):
    peso = peso
    altura = altura
    edad = edad
    sexo= sexo
    if sexo == 'H':
        TMB = 10 * int(peso) + 6.25 * int(altura) - 5 * int(edad) + 5
    elif sexo == 'M':
        TMB = 10 * int(peso) + 6.25 * int(altura) - 5 * int(edad) - 161
    else:
        print('error')
    return TMB

                
# In[9]:
actividad = st.sidebar.text_area('Cuanta actividad fisica realizas: 1-Sedentario, 2-actividad ligera, 3-actividad moderada, 4-actividad intensa, 5-actividad muy intensa', height=7)

def calculo_calorias(tmb, actividad):
    global calorias_diarias
    actividad = actividad
    if actividad == '1':
        calorias_diarias = tmb * 1.2
    elif actividad == '2':
        calorias_diarias = tmb * 1.375
    elif actividad == '3':
        calorias_diarias = tmb * 1.55
    elif actividad == '4':
        calorias_diarias = tmb * 1.725
    elif actividad == '5':
        calorias_diarias = tmb * 1.9
    else:
        print('error')
    return calorias_diarias      
    
    
dias_menu = st.sidebar.text_area('Para cuantos días desea el menú', height=7)

# In[10]:


# In[11]:





# In[12]:





# In[21]:


@st.cache

def load_data():
    data_root = "/home/dsc/TFM/repositorio/"
    columns = ['id_items', 'Nombre', 'caloria porcion','categoria']
    datafile = os.path.join(data_root, "calorias_comida.csv")
    data = pd.read_csv(datafile, sep=';', header =0, names=columns, encoding='latin-1', index_col = 0)
    return data

data = load_data()

#@st.cache

def load_data_encuesta():
    data_root= "/home/dsc/TFM/repositorio/"
    columns = ['user_id', 'rating', 'comida', 'comidas_id_org']
    datafile = os.path.join(data_root, "Encuesta_vf.csv")
    data_encuesta = pd.read_csv(datafile, sep=';', header =0, names=columns, encoding='latin-1')
    return data_encuesta

data_encuesta = load_data_encuesta()

items_id = {}
food = data_encuesta.comida.unique()
food_id = {}
items_id_orig= data_encuesta .comidas_id_org.unique()
items_id_orig_id = {}
n = 0
for x in food:
    items_id[x] = n
    food_id[n] = x
    items_id_orig_id[n]=items_id_orig[n]
    n += 1

column_items_id=[]
index = list(range(len(data_encuesta.index)))
for x in index:
    column_items_id.append(items_id[data_encuesta['comida'][x]])     
data_encuesta['items_id'] = column_items_id
# CONSULTAR al usuario

  
st.title('Cuanto te gustán las siguientes comidas')

items_id_user=[1, 2, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 95] 

            
food_user = []
for l in items_id_user:
    food_user.append(food_id[l])


rat_x0 = st.text_area(food_user[0], height=10)
rat_x1 = st.text_area(food_user[1], height=10)
rat_x2 = st.text_area(food_user[2], height=10)
rat_x3 = st.text_area(food_user[3], height=10)
rat_x4 = st.text_area(food_user[4], height=10)
rat_x5 = st.text_area(food_user[5], height=10)
rat_x6 = st.text_area(food_user[6], height=10)
rat_x7 = st.text_area(food_user[7], height=10)
rat_x8 = st.text_area(food_user[8], height=10)
rat_x9 = st.text_area(food_user[9], height=10)
rat_x10 = st.text_area(food_user[10], height=10)
rat_x11 = st.text_area(food_user[11], height=10)
rat_x12 = st.text_area(food_user[12], height=10)
rat_x13 = st.text_area(food_user[13], height=10)
rat_x14 = st.text_area(food_user[14], height=10)
rat_x15 = st.text_area(food_user[15], height=10)
rat_x16 = st.text_area(food_user[16], height=10)
rat_x17 = st.text_area(food_user[17], height=10)
rat_x18 = st.text_area(food_user[18], height=10)
rat_x19 = st.text_area(food_user[19], height=10)

rating_user = [rat_x0, rat_x1,rat_x2, rat_x3, rat_x4, rat_x5, rat_x6, rat_x7, rat_x8, 
              rat_x9, rat_x10, rat_x11, rat_x12, rat_x13, rat_x14, rat_x15, rat_x16,
              rat_x17, rat_x18, rat_x19]

food_id_orign_user = []
for l in items_id_user:
    food_id_orign_user.append(items_id_orig_id[l])

user_id_user = [data_encuesta['user_id'].max()+1]*20

df_user =pd.DataFrame()
df_user['user_id']=user_id_user
df_user['rating']=rating_user
df_user['comida']=food_user
df_user['comidas_id_org']=food_id_orign_user
df_user['items_id']=items_id_user


data_encuesta = data_encuesta.append(df_user, ignore_index=True)

n_users = data_encuesta.user_id.unique().shape[0]
n_items = data_encuesta.comida.unique().shape[0]

#####



def plan_semanal(dias_menu):
# El plan semana constara de una comida por la mañana compuesta por un plato de 2 porciones de verduras, 
# 1 porcion de proteinas y 1 porción de carbohidratos
# y como cena un plato combinado         
    
    menu_semanal = []
    comidas_utilizadas = []
    for dia in range(int(dias_menu)):
        comida_1 = []
        mix_comida_1 = []        
        calorias_1 = 0
        cena = []
        for z in use_x:
            if calorias_1 < calorias_comida:
                if 'verduras' in mix_comida_1 and 'proteinas' in mix_comida_1 and 'carbohidratos' in mix_comida_1:
                    menu_semanal.append(comida_1)
                    break

                if (data['categoria'][z]=='Verduras'):
                    if 'verduras' in mix_comida_1 or data['Nombre'][z] in comidas_utilizadas:
                        pass
                    else:
                        if calorias_1 + data['caloria porcion'][z] < calorias_comida:
                            comida_1.append(data['Nombre'][z])
                            mix_comida_1.append('verduras')
                            calorias_1 += (2*data['caloria porcion'][z])
                            comidas_utilizadas.append(data['Nombre'][z])

                if (data['categoria'][z]=='proteina'):
                    if 'proteinas' in mix_comida_1 or data['Nombre'][z] in comidas_utilizadas:
                        pass
                    else:
                        if calorias_1+ data['caloria porcion'][z] < calorias_comida:
                            comida_1.append(data['Nombre'][z])
                            mix_comida_1.append('proteinas')
                            calorias_1 += data['caloria porcion'][z]
                            comidas_utilizadas.append(data['Nombre'][z])
                if (data['categoria'][z]=='carbohidratos'):
                    if 'carbohidratos' in mix_comida_1 or data['Nombre'][z] in comidas_utilizadas:
                        pass
                    else:     
                        if calorias_1+ data['caloria porcion'][z] < calorias_comida:
                            comida_1.append(data['Nombre'][z])
                            mix_comida_1.append('carbohidratos')
                            calorias_1 += data['caloria porcion'][z]
                            comidas_utilizadas.append(data['Nombre'][z])
            else:
                menu_semanal.append(comida_1)
           
        for y in use_x:
            if len(cena)== 0:
                if (data['categoria'][y]=='Combinados'):
                    if calorias_cena * 1.2 > data['caloria porcion'][y] > calorias_cena * 0.8:
                        cena.append(data['Nombre'][y])
                        menu_semanal.append(cena)
                        comidas_utilizadas.append(data['Nombre'][y])
            
    return menu_semanal



#####De aqui en adelante voy a cambiar los train por data_encuesta

if st.button('Introducir los gustos'):
    
    uMatrixTraining = np.zeros((n_users, n_items)) # utility matrix
    for row in data_encuesta.values[:,0:5]:
        user = row[0]-1 #for use the same as index
        item = row[4]
        rating = row[1]
        uMatrixTraining[user, item] = rating    

    def cosineSimilarity(ratings, kind='user', epsilon=1e-9):
        """
        Calculate the cosine distance along the row (columns) of a matrix for users (items)

        :param ratings: a n_user X n_items matrix
        :param kind: string indicating whether we are in mode 'user' or 'item'
        :param epsilon: a small value to avoid dividing by zero (optional, defaults to 1e-9)

        :return a square matrix with the similarities
        """
        # epsilon -> small number for handling dived-by-zero errors
        if kind == 'user':
            sim = ratings.dot(ratings.T)+epsilon
        elif kind == 'item':
            sim = ratings.T.dot(ratings)+epsilon
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return sim / norms / norms.T

    itemSimilarity = cosineSimilarity(uMatrixTraining, kind='item')

    itemItemCFpredictions = uMatrixTraining.dot(itemSimilarity)/np.array([np.abs(itemSimilarity).sum(axis=1)])

    itemItemCFpredictions[uMatrixTraining>=2.0] = 0.0

    recom = itemItemCFpredictions[[data_encuesta['user_id'].max()-1],:]
    recom = np.argsort(recom)[0][0:100]

    st.write(recom.shape)
    
    comidas_recomendadas = [food_id[i] for i in recom]
    st.write(comidas_recomendadas)
    convert = [items_id_orig_id[i] for i in recom]
    

    use_x = convert
    st.write(len(use_x)) 
    st.write((use_x[0]))
    
    st.header('Requerimiento Calorico')
    st.write(calculo_calorias(calculo_tmb(peso, altura, edad, sexo), actividad))
    calorias_comida = 0.35 * calorias_diarias
    calorias_cena = 0.25 * calorias_diarias
    st.write('you need in your lunch:',calorias_comida, 'and in your dinner:',calorias_cena)
    st.header('Menu Semanal')
    st.write(plan_semanal(dias_menu))







    
    
    



