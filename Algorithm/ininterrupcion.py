import time
import keyboard

"""t=0
l = []
try:
    while True:
        time.sleep(1)
        print(t)
        t = t+1
        l.append(t)
except KeyboardInterrupt:
    pass
print(l)"""

import pandas as pd
import csv
from pandas import ExcelWriter


"""l2=[[1.2,2,3,4], [5.44, 6, 7, 8], [3,4.4,5,6,7,8,3,2]]

for i in range(len(l2)):
    #print(l2[i])
    data = np.asarray(l2[i])
    print(data)
    np.savetxt("output" + str(i) + ".csv", data.T, fmt = "%s", delimiter = ",")

    ##
    fichero = 'output' + str(i) + '.csv'
    nombre_columnas = ['Time']
    df = pd.read_csv(fichero, sep = ',', error_bad_lines = False, names = nombre_columnas)
    print(df)"""
    

import pandas
from itertools import accumulate
import numpy as np
import os


l2 = [[1,2.4,3,4], [5,6,7,8], [1,2,3,4], [4,5,5,3]]
l4 = [1,2,3,5]
l3 = [[1.2,4,5,3], [5,4,5,3], [1.4, 4, 5, 7], [5,2,3,4]]

"""l_agrupation = []

for sub_lista in range(len(l2)):
    print(l2[sub_lista])
    l_agrupation.append(l4)
    l_agrupation.append(l2[sub_lista])
print(l_agrupation)
print(len(l_agrupation))

half = int(len(l_agrupation)/2)
print(half)

length_to_split = [len(l_agrupation)//half]*half
Output = [l_agrupation[x - y: x] for x, y in zip(
          accumulate(length_to_split), length_to_split)]
print("Initial list :", l_agrupation)
print("After splitting", Output)
print(len(Output))

for p in range(len(Output)):
    print(Output[p])
    l_df = pd.DataFrame(Output[p]).transpose()
    l_df.to_csv("archivo" + str(p) + ".csv", index = False, header = False)

    fichero = 'archivo' + str(p) + '.csv'
    nombre_columnas = ['Time', 'dataA']
    df = pd.read_csv(fichero, sep = ',', error_bad_lines = False, names =nombre_columnas)
    #print(df)

instant = 2
fichero_med = 'archivo' + str(instant) + '.csv'
print(fichero_med)

nombre_columnas_med = ['Time', 'dataA']
df_med = pd.read_csv(fichero_med, sep = ',', error_bad_lines = False, names = nombre_columnas_med)
print(df_med)
lTime = list(df_med['Time'])
print(lTime)"""

"""print(".")

l_agrupacion = []

for k, l in zip(range(len(l2)), range(len(l3))):
    print(l2[k])
    print(l3[l])
    l_agrupacion.append(l4)
    l_agrupacion.append(l2[k])
    l_agrupacion.append(l3[l])
print(l_agrupacion)
print(len(l_agrupacion))

half_2 = int(len(l_agrupacion)/3)
print(half_2)

length_to_split2 = [len(l_agrupacion)//half_2]*half_2
Output2 = [l_agrupacion[x - y: x] for x, y in zip(
          accumulate(length_to_split2), length_to_split2)]
print("Initial list :", l_agrupacion)
print("After splitting", Output2)
print(len(Output2))

for p in range(len(Output2)):
    print(Output2[p])
    l_df = pd.DataFrame(Output2[p]).transpose()
    l_df.to_csv("archivo" + str(p) + ".csv", index = False, header = False)

    fichero = 'archivo' + str(p) + '.csv'
    nombre_columnas = ['Time', 'dataA', 'dataB']
    df = pd.read_csv(fichero, sep = ',', error_bad_lines = False, names =nombre_columnas)
    #print(df)

instant = 2
fichero_med = 'archivo' + str(instant) + '.csv'
print(fichero_med)

nombre_columnas_med = ['Time', 'dataA', 'dataB']
df_med = pd.read_csv(fichero_med, sep = ',', error_bad_lines = False, names = nombre_columnas_med)
print(df_med)
lTime = list(df_med['Time'])
print(lTime)"""


#file1 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_01.csv')
#file2 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_02.csv')
#file3 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_03.csv')
#file4 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_04.csv')
#file5 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_05.csv')

lista_aux = []
valor = 5
lista_aux = range(valor)
print(lista_aux)
x = list(lista_aux)
print(x)
lista_file = []
for i in range(len(x)):
    lista_file.append('file' + str(x[i]))
print(lista_file)
print(type(lista_file))

#lista_file2 = [file1, file2, file3, file4, file5]
#print(lista_file2)
#print(type(lista_file2))


"""######Concatenacion de varios archivos excel en uno solo". (IMPORTANTE PARA MODO STREAM).
import glob
path = "C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2"
csv_files = glob.glob(path + "/*.csv")

df_list = (pd.read_csv(file) for file in csv_files)
print(df_list)
#print(len(df_list))
big_df = pd.concat(df_list, ignore_index = True)    
print(big_df)
#######"""

import glob
path = "C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/measu/"

lista_aux = []
valor = 5
lista_aux = range(valor)
print(lista_aux)
x = list(lista_aux)
print(x)
lista_file = []
for i in range(len(x)):
    lista_file.append('file' + str(x[i]))
print(lista_file)
print(type(lista_file))

files = glob.glob(path + "/*.csv")
print(files)
print(type(files))

print(files[0])
name_rows_measurement = ['Time', 'Channel A', 'Channel B']
list_LTime = []
list_lChannelA = []
list_lChannelB = []
for t in range(len(files)):
    print(files[t])
    df_measu = pd.read_csv(files[t], sep = ',', error_bad_lines = False, names = name_rows_measurement)
    list_LTime.append(list(df_measu['Time']))
    list_lChannelA.append(list(df_measu['Channel A']))
    list_lChannelB.append(list(df_measu['Channel B']))
    
    #print((df_measu))   
print(len(list_LTime))
print(list_LTime[0])
print(len(list_LTime[0]))
print(len(list_lChannelA))
print(len(list_lChannelB))












    
