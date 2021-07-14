#!/usr/bin/env python
# coding: utf-8


import streamlit as st 
import numpy as np
import pandas as pd
import io
import os
import random
import plotly.graph_objects as go
from PIL import Image
import base64
from fpdf import FPDF




st.title('Genera un MENU de tu gusto y según requerimiento calórico')


image = Image.open('Metodo-plato.png')

st.image(image, use_column_width=False)

st.write("""
Un plato equilibrado incluye verduras, carbohidratos y proteinas. 
Te plantearemos un menu equilibrado que satisfaga tus gustos y tus necesidades. 

""")


peso = st.sidebar.selectbox('Ingrese su peso en Kg', list(range(40,140)))
altura = st.sidebar.selectbox('Ingrese su altura en cm', list(range(140,200))) 
edad = st.sidebar.selectbox('¿Cuántos años tiene?', list(range(16,80)))
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

                
actividad = st.sidebar.slider('Cuanta actividad fisica realizas: 1-Sedentario, 2-actividad ligera, 3-actividad moderada, 4-actividad intensa, 5-actividad muy intensa', 1, 5)


def calculo_calorias(tmb, actividad):
    global calorias_diarias
    actividad = (actividad)
    if actividad == 1:
        calorias_diarias = tmb * 1.2
    elif actividad == 2:
        calorias_diarias = tmb * 1.375
    elif actividad == 3:
        calorias_diarias = tmb * 1.55
    elif actividad == 4:
        calorias_diarias = tmb * 1.725
    elif actividad == 5:
        calorias_diarias = tmb * 1.9
    else:
        print('error')
    return calorias_diarias      
    
    
dias_menu = st.sidebar.selectbox('¿Para cuántos días desea el menú?', [1,2,3,4,5,6,7])



@st.cache

def load_data():
    data_root = "/home/dsc/TFM/repositorio/"
    columns = ['id_items', 'Nombre', 'caloria porcion','categoria']
    datafile = os.path.join(data_root, "calorias_comida.csv")
    data = pd.read_csv(datafile, sep=';', header =0, names=columns, encoding='latin-1', index_col = 0)
    return data

def load_data_link():
    data_root = "/home/dsc/TFM/repositorio/"
    columns = ['id_link', 'name_food', 'caloria porcion','link']
    datafile = os.path.join(data_root, "df_food_calories_link.csv")
    data_link = pd.read_csv(datafile, sep=',', header =0, names=columns, encoding='latin-1', index_col = 0)
    return data_link

data = load_data()
data_link = load_data_link()

data_total = data.merge(data_link, left_index=True, right_index=True)


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

  
st.title('Danos tu opnión sobre uno de los siguientes grupos de comidas')

combinacion = st.selectbox('elige un grupo', ['Grupo 1','Grupo 2','Grupo 3','Grupo 4','Grupo 5','Grupo 6','Grupo 7','Grupo 8','Grupo 9'])

if combinacion == 'Grupo 1':
    combinacion = 1
elif combinacion == 'Grupo 2':
    combinacion = 2
elif combinacion == 'Grupo 3':
    combinacion = 3
elif combinacion == 'Grupo 4':
    combinacion = 4
elif combinacion == 'Grupo 5':
    combinacion = 5
elif combinacion == 'Grupo 6':
    combinacion = 6
elif combinacion == 'Grupo 7':
    combinacion = 7
elif combinacion == 'Grupo 8':
    combinacion = 8
elif combinacion == 'Grupo 9':
    combinacion = 9


st.write('Siendo 0 No me gusta y 4 Me encanta')

items_id_user=[]
for c in range(20):
    items_id_user.append(combinacion)
    combinacion += 20


            
food_user = []
for l in items_id_user:
    food_user.append(food_id[l])


rat_x0 = st.slider(food_user[0], 0,4)
rat_x1 = st.slider(food_user[1], 0,4)
rat_x2 = st.slider(food_user[2], 0,4)
rat_x3 = st.slider(food_user[3], 0,4)
rat_x4 = st.slider(food_user[4], 0,4)
rat_x5 = st.slider(food_user[5], 0,4)
rat_x6 = st.slider(food_user[6], 0,4)
rat_x7 = st.slider(food_user[7], 0,4)
rat_x8 = st.slider(food_user[8], 0,4)
rat_x9 = st.slider(food_user[9], 0,4)
rat_x10 = st.slider(food_user[10], 0,4)
rat_x11 = st.slider(food_user[11], 0,4)
rat_x12 = st.slider(food_user[12], 0,4)
rat_x13 = st.slider(food_user[13], 0,4)
rat_x14 = st.slider(food_user[14], 0,4)
rat_x15 = st.slider(food_user[15], 0,4)
rat_x16 = st.slider(food_user[16], 0,4)
rat_x17 = st.slider(food_user[17], 0,4)
rat_x18 = st.slider(food_user[18], 0,4)
rat_x19 = st.slider(food_user[19], 0,4)

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
# El plan semanal constara de una comida por la mañana compuesta por un plato de 2 porciones de verduras, 
# 1 porcion de proteinas y 1 porción de carbohidratos
# y como cena un plato combinado         
    
    menu_semanal = []
    dias_almuerzo = []
    almuerzos = []
    link_almuerzos = []
    dias_almuerzo_pdf = []
    almuerzos_pdf = []
    link_almuerzos_pdf = []
    cenas = []
    link_cenas = []
    dias_cenas_pdf = []
    comidas_utilizadas = []
    dias = []
    cantidad = 0
    while len(almuerzos)< int(dias_menu):
        comida_1 = []
        mix_comida_1 = []
        link_individual_almuerzo = []
        calorias_1 = 0
        cena = []
        for z in use_x:
            cantidad += 1
            if calorias_1 < calorias_comida:                
                if (data['categoria'][z]=='Verduras'):
                    if 'verduras' in mix_comida_1 or data['Nombre'][z] in comidas_utilizadas:
                        continue
                    else:
                        if calorias_1 + data['caloria porcion'][z] < calorias_comida:
                            comida_2 = []
                            comida_2.append('2 Porciones ')
                            comida_2.append(data['Nombre'][z])
                            comida_3=''.join(comida_2)
                            comida_1.append(comida_3) 
                            mix_comida_1.append('verduras')
                            calorias_1 += (2*data['caloria porcion'][z])
                            comidas_utilizadas.append(data['Nombre'][z])
                            link_individual_almuerzo.append(data_total['link'][z])
                                                       

                if (data['categoria'][z]=='proteina'):
                    if 'proteinas' in mix_comida_1 or data['Nombre'][z] in comidas_utilizadas:
                        continue
                    else:
                        if calorias_1+ data['caloria porcion'][z] < calorias_comida:
                            comida_2 = []
                            comida_2.append('1 Porción ')
                            comida_2.append(data['Nombre'][z])
                            comida_3=''.join(comida_2)
                            comida_1.append(comida_3)
                            mix_comida_1.append('proteinas')
                            calorias_1 += data['caloria porcion'][z]
                            comidas_utilizadas.append(data['Nombre'][z])
                            link_individual_almuerzo.append(data_total['link'][z])
                            
                            
                if (data['categoria'][z]=='carbohidratos'):
                    if 'carbohidratos' in mix_comida_1 or data['Nombre'][z] in comidas_utilizadas:
                        continue
                    else:     
                        if calorias_1+ data['caloria porcion'][z] < calorias_comida:
                            comida_2 = []
                            comida_2.append('1 Porción ')
                            comida_2.append(data['Nombre'][z])
                            comida_3=''.join(comida_2)
                            comida_1.append(comida_3)
                            mix_comida_1.append('carbohidratos')
                            calorias_1 += data['caloria porcion'][z]
                            comidas_utilizadas.append(data['Nombre'][z])
                            link_individual_almuerzo.append(data_total['link'][z])
                            
                            
                if 'verduras' in mix_comida_1 and 'proteinas' in mix_comida_1 and 'carbohidratos' in mix_comida_1:
                    comida_4='<br>'.join(comida_1)
                    menu_semanal.append(comida_1)
                    almuerzos.append(comida_4)
                    comida_pdf="\n".join(comida_1)
                    almuerzos_pdf.append(comida_pdf)
                    link_pdf ="\n".join(link_individual_almuerzo)
                    link_almuerzos_pdf.append(link_pdf)
                    link_2 ='<br>'.join(link_individual_almuerzo)
                    link_almuerzos.append(link_2)
                    dias_pdf = []
                    dias_pdf.append('Almuerzo Día ')
                    dias_pdf.append(str(len(almuerzos)))
                    dias_pdf2 = ''.join(dias_pdf)
                    dias_almuerzo_pdf.append(dias_pdf2)
                    dias.append(len(almuerzos))                    
                    break

                if 'verduras' in mix_comida_1 and calorias_1 > 0.8*calorias_comida:
                    comida_4='<br>'.join(comida_1)
                    menu_semanal.append(comida_1)
                    almuerzos.append(comida_4)
                    link_2 ='<br>'.join(link_individual_almuerzo)
                    link_almuerzos.append(link_2)
                    dias.append(len(almuerzos)) 
                    comida_pdf="\n".join(comida_1)
                    almuerzos_pdf.append(comida_pdf)
                    link_pdf ="\n".join(link_individual_almuerzo)
                    link_almuerzos_pdf.append(link_pdf)
                    dias_pdf = []
                    dias_pdf.append('Almuerzo Día ')
                    dias_pdf.append(str(len(almuerzos)))
                    dias_pdf2 = ''.join(dias_pdf)
                    dias_almuerzo_pdf.append(dias_pdf2)
                    break
             
            else:
                comida_4='<br>'.join(comida_1)
                menu_semanal.append(comida_1)
                almuerzos.append(comida_4)                
                link_2 ='<br>'.join(link_individual_almuerzo)
                link_almuerzos.append(link_2)               
                dias.append(len(almuerzos))
                comida_pdf="\n".join(comida_1)
                almuerzos_pdf.append(comida_pdf)
                link_pdf ="\n".join(link_individual_almuerzo)
                link_almuerzos_pdf.append(link_pdf)
                dias_pdf = []
                dias_pdf.append('Almuerzo Día ')
                dias_pdf.append(str(len(almuerzos)))
                dias_pdf2 = ''.join(dias_pdf)
                dias_almuerzo_pdf.append(dias_pdf2)
                break


    while len(cena)< int(dias_menu):                
        for y in use_x:
            if data['categoria'][y]=='Combinados' and data['Nombre'][y] not in comidas_utilizadas:
                if calorias_cena * 1.2 > data['caloria porcion'][y] > calorias_cena * 0.8:
                    cena.append(data['Nombre'][y])
                    menu_semanal.append(cena)
                    comidas_utilizadas.append(data['Nombre'][y])
                    cenas.append(data['Nombre'][y])
                    link_cenas.append(data_total['link'][y])
                    dias_cena_pdf = []
                    dias_cena_pdf.append('Cena Día ')
                    dias_cena_pdf.append(str(len(cena)))
                    dias_cena_pdf2 = ''.join(dias_cena_pdf)
                    dias_cenas_pdf.append(dias_cena_pdf2)
                    break     
        
    
    df_almuerzos = pd.DataFrame({'Día':dias, 'Almuerzos':almuerzos, 'Recetas':link_almuerzos})
    data_pdf = []
    for a, b, c in zip(dias_almuerzo_pdf, almuerzos_pdf, link_almuerzos_pdf):
        lista = [a,b,c]
        data_pdf.append(lista)
    
    data_cena_pdf = []
    for a, b, c in zip(dias_cenas_pdf, cenas, link_cenas):
        lista = [a,b,c]
        data_cena_pdf.append(lista)
    
    df_cenas = pd.DataFrame({'cenas':cenas, 'link cenas':link_cenas})    
    fig = go.Figure(data=[go.Table(columnwidth=[80,400,400],header=dict(values=['Día','Almuerzo', 'Recetas']), cells=dict(values = [dias,almuerzos,link_almuerzos], align=['center', 'left', 'left']))])
    fig.update_layout(margin_b = 0, margin_l= 0,width=1000, height=600)
    fig_cena = go.Figure(data=[go.Table(columnwidth=[80,400,400],header=dict(values=['Día','Cena', 'Recetas']), cells=dict(values = [dias,cenas,link_cenas], align=['center', 'center', 'left']))])
    fig_cena.update_layout(margin_t = 0, margin_l= 0, width=1000, height=400)
    
    return df_almuerzos, fig_cena, fig, data_pdf, data_cena_pdf




def create_download_link(val, filename):
    """ Generate a link to be download"""
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Descarga tu Menú </a>'



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
    recom = np.argsort(recom)[0][0:200]

    
    
    comidas_recomendadas = [food_id[i] for i in recom]
    
    convert = [items_id_orig_id[i] for i in recom]
    use_x = convert

    
    st.header('Requerimiento Calorico')
    st.write(round(calculo_calorias(calculo_tmb(peso, altura, edad, sexo), actividad)))
    calorias_comida = round(0.35 * calorias_diarias)
    calorias_cena = round(0.28 * calorias_diarias)
    st.write('Debes ingerir en la comida:',calorias_comida, 'calorias y en la cena:',calorias_cena, 'calorias')
    st.header('Menu Semanal')
    df, fig_cena, fig, data_pdf, data_cena_pdf = plan_semanal(dias_menu)

    st.write(fig)
    st.write(fig_cena)


    def export_as_pdf():    
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Times', '', 8)
        
        epw = pdf.w - 3*pdf.l_margin
        col_width = epw/2
        
        data = data_pdf
        th = pdf.font_size
        
        for row in data:
            for datum in row:
                if datum == row[0]:
                    pdf.set_fill_color(135, 206, 235)
                    pdf.multi_cell(col_width, 2*th, str(datum), border=1, align = 'C', fill = True)                      
                else:
                    pdf.multi_cell(col_width, 2*th, str(datum), border=1, align='C')
            pdf.ln(2*th)
        for row in data_cena_pdf:
            for datum in row:
                if datum == row[0]:
                    pdf.set_fill_color(143, 237, 143)
                    pdf.multi_cell(col_width, 2*th, str(datum), border=1, align = 'C', fill = True)                      
                else: 
                    pdf.multi_cell(col_width, 2*th, str(datum), border=1, align = 'C')   
            pdf.ln(2*th)

        html = create_download_link(pdf.output(dest='S').encode('Latin-1'),"test")

        return st. markdown(html, unsafe_allow_html=True)  
    
    export_as_pdf()
    






    
    
    



