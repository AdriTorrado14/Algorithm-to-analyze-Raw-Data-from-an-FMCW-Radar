from __future__ import division
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.fft
from intersect import intersection
from scipy.signal import find_peaks

import statistics
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import blackmanharris, correlate, chirp, peak_widths
from time import time
import sys
import itertools

from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
#from parabolic import parabolic

#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_01.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_02.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_03.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_04.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_05.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_06.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_07.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_08.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_09.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_10.csv')

#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_01.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_02.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_03.csv')
#filename = (C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_04.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_05.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_06.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_07.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_08.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_09.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_10.csv')

#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_01.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_02.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_03.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_04.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_05.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_06.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_07.csv')
filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_08.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_09.csv')
#filename = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_10.csv')

data = pd.read_csv(filename, header = 0)

columns_names = data.columns.values

#########################################################################
#Data processing of the excel file.
listTime = list(data['Time'])
listAuxTime = listTime.copy()
listAuxTime.pop(0)
listAuxTimeReal = list(float(i) for i in listAuxTime)
#print(len(listAuxTimeReal))

listChannelA = list(data['Channel A'])
listAuxChannelA = listChannelA.copy()
listAuxChannelA.pop(0)
listAuxChannelAReal = list(float(i) for i in listAuxChannelA)

listChannelB = list(data['Channel B'])
listAuxChannelB = listChannelB.copy()
listAuxChannelB.pop(0)
listAuxChannelBReal = list(float(i) for i in listAuxChannelB)

#listAuxChannelB_Real_corr100 = list(map(lambda x : x/10, listAuxChannelBReal))

#########################################################################

#Frequency, samples and period.
intervalTime = abs(listAuxTimeReal[10]-listAuxTimeReal[9])
#print("Interval:" + str(intervalTime))
intervalTime_seg = intervalTime/1000
#print("Interval en seg: " + str(intervalTime_seg))
freq = 20 #Frequency
T = 1/freq #Period
T_halfPeriod = T/2 #Half period
frequency_Sampling = 1/intervalTime_seg #Frequency sampling.
#print(frequency_Sampling)
#print("Frequency sampling: " + str(frequency_Sampling))
intervalT_halfPeriod = intervalTime_seg + T_halfPeriod
#print("Half period + interval: " + str(intervalT_halfPeriod))
samples = T_halfPeriod / intervalTime_seg
#print("Samples:" + str(samples))
samplesRound = round(samples)
#print("Sample rouding: " + str(samplesRound))
samplesRound_half = samplesRound/2
#print("Half of the samples:" + str(samplesRound_half))
samplesRound_half_Round = round(samplesRound_half)+1
#print("Rounding of half of the samples.: " + str(samplesRound_half_Round))
#########################################################################

#List processing to catch half-period signals.
lTime = []
lTime = listAuxTimeReal.copy()
print("-------------------------------------------------")
#print(len(lTime))
#del lTime[muestraR+77:len(listAuxTimeReal)]
del lTime[(samplesRound*2)+77:len(listAuxTimeReal)] #Para coger medio periodo en ascenso
#print(len(lTime))
lTime_aux = []
lTime_aux = lTime.copy()
del lTime_aux[0:230] # Desde 0 hasta la muestra 230
#print(len(lTime_aux))

lChannelA = []
lChannelA = listAuxChannelAReal.copy()
#del lChannelA[muestraR+77:len(listAuxChannelAReal)]
del lChannelA[(samplesRound*2)+77:len(listAuxChannelAReal)]
lChannelA_aux = []
lChannelA_aux = lChannelA.copy()
del lChannelA_aux[0:230]
#print(len(lChannelA_aux))

lChannelB = []
lChannelB = listAuxChannelBReal.copy()
#print(len(lChannelB))
#del lChannelB[muestraR+77:len(listAuxChannelBReal)]
del lChannelB[(samplesRound*2)+77:len(listAuxChannelBReal)]
lChannelB_aux = []
lChannelB_aux = lChannelB.copy()
del lChannelB_aux[0:230]
#print(len(lChannelB_aux))
#########################################################################

#########################################################################
#Polynomial Approximation.
print("----Polynomial Approximation----")
modelo = np.poly1d(np.polyfit(lTime_aux, lChannelB_aux, 3))
#print(modelo)
#print(len(modelo))
polyline = np.linspace(lTime_aux[0],lTime_aux, len(lTime_aux))
#polyline = np.linspace(lTime_aux[0],lTime_aux,1)
#print(polyline)
#print(len(polyline))

listModel_poly = []
listModel_poly = modelo(polyline)
#print(listModel_poly)
#print(len(listModel_poly))
#print(len(listModel_poly[0]))
agrupation_listModel_poly = listModel_poly[len(lTime_aux)-1]
#print(len(lTime_aux))
#agrupation_listModel_poly2 = listModel_poly[0]
#print(agrupation_listModel_poly)
#print(agrupation_listModel_poly2)
#print(len(agrupation_listModel_poly))
list_agrupation_listModel_poly = agrupation_listModel_poly.tolist() #list conversion.
subtract_list = [t1-t2 for t1, t2 in zip(lChannelB_aux, list_agrupation_listModel_poly)]
#print(subtract_list)
#print(type(subtract_list))
print("----End Polynomial Approximation----")
#########################################################################

#Intersection half period.
print("----INTERSECTION----")
x_axisR, y_axisR = intersection(lTime_aux, lChannelA_aux, lTime_aux, lChannelB_aux) #Intersection original signal (ChannelB)
x_axisP,y_axisP = intersection(lTime_aux, lChannelA_aux, lTime_aux, subtract_list) #Intersection corrected signal (polynomial)
#print("Intersection X-axis.")
#print(x_axisP)
#print(len(x_axisP))
#print(type(x_axisP))
#print("Intersection Y-axis.")
#print(y_axisP)
#print(len(y_axisP))
#print(type(y_axisP))
x_axisP_list = []
x_axisP_list = x_axisP.tolist() #Polynomial
#print(x_axisP_list)
#print(len(x_axisP_list))

#print("List Intersection X-axis.")
#print(x_axisP_list)
y_axisP_list = []
y_axisP_list = y_axisP.tolist()
#print("List Intersection Y-axis.")
#print(y_axisP_list)

#Part of the original signal.
x_axisR_list = []
x_axisR_list = x_axisR.tolist()
#print(x_axisR_list)
#print(len(x_axisR_list))
y_axisR_list = []
y_axisR_list = y_axisR.tolist()
print("----END INTERSECTION----")
#########################################################################

#########################################################################
print("----FIND PEAKS----")
#First version for 1 meter.
peaks, _ = find_peaks(listAuxChannelAReal, height = 7.8)
list_peaks = []
list_peaks = peaks.tolist()

peaks_list_lTime = []
for s in list_peaks:
    peaks_list_lTime.append(listAuxTimeReal[s])
print(peaks_list_lTime)

list_peaks_func = []
list_peaks_func = list_peaks.copy()
#print(list_peaks_func)

peaks_list_lChannelA = []
for d in list_peaks:
    peaks_list_lChannelA.append(listAuxChannelAReal[d])
#print(len(peaks_list_lChannelA))
#print(peaks_list_lChannelA)

peaks_B, _ = find_peaks(listAuxChannelBReal, height = 4, distance = 15)
list_peaksB = []
list_peaksB = peaks_B.tolist()
#print(list_peaksB)
#print(len(list_peaksB))

peaks_list_channelB = []
for g in list_peaksB:
    peaks_list_channelB.append(listAuxChannelBReal[g])
#print(peaks_list_channelB)
#print(len(peaks_list_channelB))

peaks_list_lTime_B = []
for q in list_peaksB:
    peaks_list_lTime_B.append(listAuxTimeReal[q])
#print(peaks_list_lTime_B)
#print(len(peaks_list_lTime_B))

print("----END FIND PEAKS----")

###########################################################################################

l_indices = []
for t in range(len(peaks_list_lTime)):
    #print(peaks_list_lTime[t])
    l_indices.append(listAuxTimeReal.index(peaks_list_lTime[t]))

#print(l_indices)
#print(listAuxTimeReal[l_indices[0]])
#print(listAuxTimeReal[l_indices[1]])
#print(listAuxTimeReal[l_indices[2]])
#print(listAuxTimeReal[l_indices[3]])

valor = 0
for g in range(len(l_indices)):
    #print(l_indices[g])
    valor = l_indices[g]+samplesRound
    #print(valor)
    l_indices.append(valor)

#print(l_indices)
l_indices.sort()
#print(l_indices)
l_indices.pop(0)
l_indices.pop(len(l_indices)-1)
#print(l_indices)
print("-")

#####SEPARACION#####
x = 2 #Separacion de 2.
final_list = lambda l_indices, x: [l_indices[i:i+x] for i in range(0, len(l_indices), x)]
output = final_list(l_indices, x)
#print(output)
#print(len(output))

#####PERIODOS A ANALIZAR#####
list_period_Time = []
list_period_ChannelA = []
list_period_ChannelB = []
for j in range(len(output)):
    #print(output[j])
    list_period_Time.append(listAuxTimeReal[output[j][0]:output[j][1]])
    list_period_ChannelA.append(listAuxChannelAReal[output[j][0]:output[j][1]])
    list_period_ChannelB.append(listAuxChannelBReal[output[j][0]:output[j][1]])

print(".")
print("time")
#print(len(list_period_Time))
print((list_period_Time[0]))
print(len(list_period_Time[0]))
print(len(list_period_Time[1]))
print(len(list_period_Time[2]))
print("channelA")
print(len(list_period_ChannelA[0]))
#print(len(list_period_ChannelA[1]))
#print(len(list_period_ChannelA[2]))
print("ChannelB")
print(len(list_period_ChannelB[0]))
#print(len(list_period_ChannelB[1]))
#print(len(list_period_ChannelB[2]))
######---######

######POLYNOMIAL#####
print("--------------POLINOMIAL----------------")
modelo_full = []
polyline_full = []

#modelo_f = np.poly1d(np.polyfit(list_period_Time[0], list_period_ChannelB[0], 
for t, h in zip(range(len(list_period_Time)), range(len(list_period_ChannelB))):
    modelo_full.append(np.poly1d(np.polyfit(list_period_Time[t], list_period_ChannelB[h],3)))
    polyline_full.append(np.linspace(list_period_Time[t][0], list_period_Time[t], len(list_period_Time[t])))
print("modelo")
#print(modelo_full)
#print(len(modelo_full))
#print(len(modelo_full[0]))
#print(len(modelo_full[1]))
print("Poly-")
#print(polyline_full)
#print(len(polyline_full))
#print(len(polyline_full[0]))
#print(len(polyline_full[1]))
#print(len(polyline_full[2]))

print("-")
listModel_poly_full = []
for g, m in zip(range(len(modelo_full)), range(len(polyline_full))):
    listModel_poly_full.append(modelo_full[g](polyline_full[m]))

#print(listModel_poly_full)
#print(len(listModel_poly_full))
#print(len(listModel_poly_full[0]))
#print(len(listModel_poly_full[1]))
#print(len(listModel_poly_full[2]))
print(".")


#####HACER AUTOMATICO
agrupation_listModel_poly_full = []
agrupation_listModel_poly_full.append(listModel_poly_full[0][len(listModel_poly_full[0])-1])
agrupation_listModel_poly_full.append(listModel_poly_full[1][len(listModel_poly_full[1])-1])
agrupation_listModel_poly_full.append(listModel_poly_full[2][len(listModel_poly_full[2])-1])

#####AUTOMATICO -> DONE.
agrupation_listModel_poly_full2 = []
for g in range(len(listModel_poly_full)):
    agrupation_listModel_poly_full2.append(listModel_poly_full[g][len(listModel_poly_full[g])-1])

print("agrupation2")
#print(agrupation_listModel_poly_full2)
#print(len(agrupation_listModel_poly_full2))
#print(len(agrupation_listModel_poly_full2[0]))
#print(len(agrupation_listModel_poly_full2[1]))
#print(len(agrupation_listModel_poly_full2[2]))

print("agrupation")
#print(agrupation_listModel_poly_full)
#print(len(agrupation_listModel_poly_full))
#print(len(agrupation_listModel_poly_full[0]))
#print(len(agrupation_listModel_poly_full[1]))
#print(len(agrupation_listModel_poly_full[2]))

subtract_list_full = []
for g,f in zip(range(len(list_period_ChannelB)), range(len(agrupation_listModel_poly_full))):
    subtract_list_full.append([t1 - t2 for t1, t2 in zip(list_period_ChannelB[g], agrupation_listModel_poly_full[f])])
print("subtract")
#print(subtract_list_full)
#print(len(subtract_list_full))
#print(len(subtract_list_full[0]))
#print(subtract_list_full[0])
print("-----3")
#print(subtract_list_full[1])
#print(len(subtract_list_full[2]))

#####Zero cross without flat.
print("-")
zero_cross_full = []
for h in range(len(subtract_list_full)):
    zero_cross_full.append(np.where(np.diff(np.sign(subtract_list_full[h])))[0])
print("zero cross")
#print(zero_cross_full)
#print(len(zero_cross_full))
#print(len(zero_cross_full[0]))
#print(len(zero_cross_full[1]))
#print(len(zero_cross_full[2]))

l_pol = []
for d in range(len(zero_cross_full)):
    l_pol.append(zero_cross_full[d]+1)
print("list pol")
#print(l_pol)
#print(len(l_pol[0]))
#print(len(l_pol[1]))
#print(len(l_pol[2]))

#####################################HACER AUTOMATICO#######################################
##### AUTOMATICO --> DONE
#FIRST PERIOD
kil_1 = []
kil_1.append(zero_cross_full[0])
kil_1.append(l_pol[0])
kil_1f = [item for sublist in kil_1 for item in sublist]
#print(kil_1f)
kil_1f.sort()
#print(kil_1f)
#SECOND PERIOD
kil_2 = []
kil_2.append(zero_cross_full[1])
kil_2.append(l_pol[1])
kil_2f = [item for sublist in kil_2 for item in sublist]
kil_2f.sort()
#THIRD PERIOD
kil_3 = []
kil_3.append(zero_cross_full[2])
kil_3.append(l_pol[2])
kil_3f = [item for sublist in kil_3 for item in sublist]
kil_3f.sort()
#UNION
kil_def = []
kil_def.append(kil_1f)
kil_def.append(kil_2f)
kil_def.append(kil_3f)
print("kildef")
#print(kil_def)
#print(len(kil_def))
zero_cross_def_full = []
zero_cross_def_full = kil_def.copy()
print(zero_cross_def_full)
print(len(zero_cross_def_full[0]))
print(zero_cross_def_full[0])

###
print("automatico")
kij = []
for t,g in zip(range(len(zero_cross_full)), range(len(l_pol))):
    kij.append(zero_cross_full[t])
    kij.append(l_pol[g])
    kij1 = [item for sublist in kij for item in sublist]
print(kij1)
print(kij)

x2 = 2*len(l_pol[0]) #Separacion de 2.
final_list_r = lambda kij1, x2: [kij1[i:i+x2] for i in range(0, len(kij1), x2)]
outputr = final_list(kij1, x2)
print(outputr)
print(len(outputr))
for j in range(len(outputr)):
    outputr[j].sort()
print(outputr)
print(len(outputr))
print(len(outputr[0]))
print(len(outputr[1]))
print(len(outputr[2]))


print(subtract_list_full)
print(len(subtract_list_full[0]))
print(len(subtract_list_full[1]))
print(len(subtract_list_full[2]))
####################################HACER AUTOMATICO --> DONE########################################

#####INTERPOLATION - FIRST PART - GET POINTS OF ZERO CROSS.
#####HACER AUTOMATICO --> DONE
pos_subtractSignal_full_def2 = []
for t in range(len(outputr)):
    for f in range(len(outputr[t])):
        try:
            pos_subtractSignal_full_def2.append(subtract_list_full[t][outputr[t][f]])
        except IndexError:
            pass
print("Desbordamiento")
print(pos_subtractSignal_full_def2)
print(len(pos_subtractSignal_full_def2))


#UNION EN SUBLISTA DE LONGITUD 3
g = len(outputr[0])
final_list_rrp = lambda pos_subtractSignal_full_def2, g: [pos_subtractSignal_full_def2[i:i+g] for i in range(0, len(pos_subtractSignal_full_def2), g)]
outputrg = final_list_rrp(pos_subtractSignal_full_def2, g)
print(outputrg)
print(len(outputrg))

pos_Time_full_def = []
for h in range(len(outputr)):
    for l in range(len(outputr[h])):
        try:
            pos_Time_full_def.append(list_period_Time[h][outputr[h][l]])
        except IndexError:
            pass

#UNION EN SUBLISTA DE LONGITUD 3
final_list_rrpTime = lambda pos_Time_full_def, g: [pos_Time_full_def[i:i+g] for i in range(0, len(pos_Time_full_def), g)]
outputrg_Time = final_list_rrpTime(pos_Time_full_def, g)
print(outputrg_Time)
print(len(outputrg_Time))
print(len(outputrg_Time[0]))
print(len(outputrg_Time[1]))
print(len(outputrg_Time[2]))
print("Comprobacion desbordamiento")

#####INTERPOLATION CALCULUS
###############################HACER AUTOMATICO --> DONE
list_Interpolation_ZC = []
for t,g in zip(range(len(outputrg)), range(len(outputrg_Time))):
    for d,h in zip(range(len(outputrg[t])-1), range(len(outputrg_Time[g])-1)):
        try:
            pointst1 = outputrg_Time[g][h] - ((outputrg[t][d]/(outputrg[t][d+1] - outputrg[t][d]))*(outputrg_Time[g][h+1]-outputrg_Time[g][h]))
        except ZeroDivisionError:
           pointst1 = 0
        list_Interpolation_ZC.append(pointst1)

#UNION EN SUBLISTA DE LONGITUD 3
#print(list_Interpolation_ZC)
f = len(outputrg[0])-1
f_interpolation = lambda list_Interpolation_ZC, f:[list_Interpolation_ZC[i:i+f] for i in range(0, len(list_Interpolation_ZC), f)]
output_f_interpolation = f_interpolation(list_Interpolation_ZC, f)
#print(output_f_interpolation)
#print(len(output_f_interpolation))

for d in range(len(output_f_interpolation)):
    del output_f_interpolation[d][1:len(output_f_interpolation[d]):2]
#print(output_f_interpolation)

#########TO GET THE COMPLETE PERIOD##########
for t in range(len(output_f_interpolation)):
    del output_f_interpolation[t][1:len(output_f_interpolation[t]):2]
#print(output_f_interpolation)

###################TO CHECK IF WE TAKE THE CORRECT DATA##################
list_proof = []
for g in range(len(output_f_interpolation[1])-1):
    list_proof.append(output_f_interpolation[1][g+1]-output_f_interpolation[1][g])
#print(list_proof)
#########################################################################

###ALL AT THE SAME TIME
list_proff = []
for g in range(len(output_f_interpolation)):
    for d in range(len(output_f_interpolation[g])-1):
        list_proff.append(output_f_interpolation[g][d+1]-output_f_interpolation[g][d])
#print(list_proff)

average_periods_interpolation_tog = statistics.mean(list_proff)
#print(average_periods_interpolation_tog)

#####LIST'S DIVISION BY 3
fir = len(output_f_interpolation[0])-1
f_periods_interpolation = lambda list_proff, fir:[list_proff[i:i+fir] for i in range(0, len(list_proff), fir)]
output_f_periods_interpolation = f_periods_interpolation(list_proff, fir)
print(output_f_periods_interpolation)
print(len(output_f_periods_interpolation[0]))
print(len(output_f_periods_interpolation[1]))
print(len(output_f_periods_interpolation[2]))
print("-")

####HACER AUTOMATICO --> DONE
average_periods_interpolation = []
sum_period = []
for t in range(len(output_f_periods_interpolation)):
    average_periods_interpolation.append(statistics.mean(output_f_periods_interpolation[t]))
    sum_period.append(sum(output_f_periods_interpolation[t]))
print(average_periods_interpolation)
print(sum_period)

###DISTANCE CALCULUS.
frequency_estimation_ZC = []
for r,g in zip(range(len(average_periods_interpolation)), range(len(sum_period))):
    frequency_estimation_ZC.append(0.5*(len(output_f_periods_interpolation[g])/sum_period[g]*1000))
print(frequency_estimation_ZC)

speed_ligth = 3e8
Freq_mod = 20
Period_mod = 1/Freq_mod
Period_half_mod = Period_mod/2
point_Freq_high = 24.24e9
point_Freq_low = 24.065e9
deltaFreq = point_Freq_high - point_Freq_low
#deltaFreq = 150e6

R_ZC_all_together = []
for t in range(len(frequency_estimation_ZC)):
    R_ZC_all_together.append((speed_ligth*frequency_estimation_ZC[t]*Period_mod)/(2*2*deltaFreq))
print(R_ZC_all_together)
###--> DONE
#########################################################################


#########################################################################
#Process of Cross-Zero crossing.
print("----Begin Zero Crossings----")
zero_cross = np.where(np.diff(np.sign(subtract_list)))[0]
#print(zero_cross)
zero_crossN = []
zero_crossN = zero_cross.tolist()
#print(zero_crossN)
#print(subtract_list)
#print(lTime_aux)

zero_cross_CB = np.where(np.diff(np.sign(lChannelB_aux)))[0]
zero_crossN_CB = []
zero_crossN_CB = zero_cross_CB.tolist()
#print(zero_crossN_CB)
#print("----------------------------------------------------")
listAux_zeroCross_Poly = []
for i in zero_crossN:
    listAux_zeroCross_Poly.append(subtract_list[i])#Positions of zero crossings in corrected signal.
#print("Positions of zero crossings in corrected signal.")
#print(listAux_zeroCross_Poly)
#print("-----------------------------------------------------")
listAux_zeroCross_lTime = []
for j in zero_crossN:
    listAux_zeroCross_lTime.append(lTime_aux[j]) #Positions of zero crossings in lTime.
print("Positions of zero crossings in lTime.")
#print(listAux_zeroCross_lTime)
#print(len(listAux_zeroCross_lTime))
#print("-----------------------------------------------------")

#### Procesado para ChannelB original
l_aux_zeroCross_CB = []
for g in zero_crossN_CB:
    l_aux_zeroCross_CB.append(lChannelB_aux[g])

l_aux_zeroCross_Time_CB = []
for f in zero_crossN_CB:
    l_aux_zeroCross_Time_CB.append(lTime_aux[f])
#print(l_aux_zeroCross_Time_CB)
#### Procesado para ChannelB original.

listAux_lTime_subtractIndex = []
for m in range(len(listAux_zeroCross_lTime)-1):
    listAux_lTime_subtractIndex.append(abs(listAux_zeroCross_lTime[m+1]-listAux_zeroCross_lTime[m]))
print("Zero Ltime crossings:")
#print(listAux_lTime_subtractIndex)
#print(len(listAux_lTime_subtractIndex))

sum_listAux_lTime_subIndex = sum(listAux_lTime_subtractIndex)
#print(sum_listAux_lTime_subIndex)
#print("-----------------------------------------------------")
#print("--------")
#sum_list_lTime_subIndex_2 = sum(listAux_lTime_subtractIndex) #It should be the same value than "sum_listAux_lTime_subIndex".
#print("Sum of times corresponding to zero crossings: " + str(sum_list_lTime_subIndex_2))
print("----End Zero Crossings----")

#### Procesado para ChannelB original
lAux_Time_CB = []
for t in range(len(l_aux_zeroCross_Time_CB)-1):
    lAux_Time_CB.append(abs(l_aux_zeroCross_Time_CB[t+1]-l_aux_zeroCross_Time_CB[t]))
#print(lAux_Time_CB)

sum_laux_Time_subIndex = sum(lAux_Time_CB)
#print(sum_laux_Time_subIndex)

mean_laux_Time = statistics.mean(lAux_Time_CB)
#print(mean_laux_Time)
#### Procesado para ChannelB original

#########################################################################

#########################################################################
print("----Begin Interpolation----")
#Interpolation Zero Crossing Method.
zero_crossN_aux = []
zero_crossN_aux = zero_crossN.copy()
#print(zero_crossN)
#print(zero_crossN_aux)
list_zeroCrossN_aux = []

for h in range(len(zero_crossN)):
    list_zeroCrossN_aux.append(zero_crossN[h]+1)
    zero_crossN.append(zero_crossN[h]+1)
    i = i+1
#print(list_zeroCrossN_aux)
#print(zero_crossN)
zero_crossN.sort()
#print(zero_crossN) #Lista ordenada

list_lChannelB_aux_ZC = []
for p in zero_crossN:
    list_lChannelB_aux_ZC.append(subtract_list[p])
#print(list_lChannelB_aux_ZC)
#print(len(list_lChannelB_aux_ZC))

list_lTime_aux_ZC = []
for q in zero_crossN:
    list_lTime_aux_ZC.append(lTime_aux[q])
#print(list_lTime_aux_ZC)
#print(len(list_lTime_aux_ZC))

list_Interpolation_ZC = []
for k, l in zip(range(len(list_lChannelB_aux_ZC)-1), range(len(list_lTime_aux_ZC)-1)):
    try:
        point_t1 = list_lTime_aux_ZC[l]-((list_lChannelB_aux_ZC[k]/(list_lChannelB_aux_ZC[k+1]-list_lChannelB_aux_ZC[k]))*(list_lTime_aux_ZC[l+1]-list_lTime_aux_ZC[l]))
    except ZeroDivisionError:
        point_t1=0
    #print(point_t1)
    list_Interpolation_ZC.append(point_t1)
#print(list_Interpolation_ZC)
#print(len(list_Interpolation_ZC))
del list_Interpolation_ZC[1:len(list_Interpolation_ZC):2]
#print(list_Interpolation_ZC)
#print(len(list_Interpolation_ZC))

list_Interpolation_ZC_P = []
list_Interpolation_ZC_P = list_Interpolation_ZC.copy()
#print(list_Interpolation_ZC_P)
del list_Interpolation_ZC_P[1::2]
#print(list_Interpolation_ZC_P)

list_Interp_ZC_P_def = []
for v in range(len(list_Interpolation_ZC_P)-1):
    list_Interp_ZC_P_def.append(list_Interpolation_ZC_P[v+1]-list_Interpolation_ZC_P[v])
#print(list_Interp_ZC_P_def)
#print(len(list_Interp_ZC_P_def))
sum_Interpolation = sum(list_Interp_ZC_P_def)
#print(sum_Interpolation)

print("----End Interpolation----")
#########################################################################

#########################################################################
#Variant: Average.
mean_interpolation_ZC = statistics.mean(list_Interp_ZC_P_def)
#print("Average value interpolation: " + str(mean_interpolation_ZC))

#Another way
value_time_interpolation_ZC = list_Interpolation_ZC_P[len(list_Interpolation_ZC_P)-1]-list_Interpolation_ZC_P[0]
#print("Last sample (time) - first sample (time): " + str(value_time_interpolation_ZC))

period_time_interpolation_ZC = value_time_interpolation_ZC/len(list_Interpolation_ZC_P)
#print(period_time_interpolation_ZC)

#Second variant: Average between mean_interpolation_ZC and period_time_interpolation_ZC
average_secondVariant = (mean_interpolation_ZC + period_time_interpolation_ZC)/2
#print(average_secondVariant)
#########################################################################

#########################################################################
##Different methods for distance calculation.
#Zero crossing.
print("----Begin Distance Calculation----")
speed_ligth = 3e8
Freq_mod = 20
Period_mod = 1/Freq_mod
Period_half_mod = Period_mod/2
point_Freq_high = 24.24e9
point_Freq_low = 24.065e9
deltaFreq = point_Freq_high - point_Freq_low
#deltaFreq = 150e6

print("----BEGIN DISTANCE ZC----")
#distance_ZC_meanInterpolation =

#distance_ZC_numbwave =

#distance_mean_both = 

print("----END DISTANCE ZC----")

#Distance
if len(x_axisP_list) <= 5:
    print("Caso 1 meter")
    lista_caso1 = []
    for kl in range(len(peaks_list_lTime)-1):
        lista_caso1.append(peaks_list_lTime[kl+1]-peaks_list_lTime[kl])
    list_value_T = statistics.mean(lista_caso1)
    value_freq = 1/list_value_T
    value_freq_n = value_freq*1000
    R2 = (speed_ligth*value_freq_n*Period_mod)/(2*2*deltaFreq)
    print("Distance for first case: " + str(R2))

#########################################################################
list_frequencies_ZC = []
#Last version.
print("----START ZERO CROSSING-DISTANCE----")
copy_listAux_zeroCross_lTime = []
#print(listAux_zeroCross_lTime)
#print(len(listAux_zeroCross_lTime))
copy_listAux_zeroCross_lTime = listAux_zeroCross_lTime.copy()
del copy_listAux_zeroCross_lTime[1::2]
#print(copy_listAux_zeroCross_lTime)

listAux_sumProcess = []
for h in range(len(copy_listAux_zeroCross_lTime)-1):
    listAux_sumProcess.append(copy_listAux_zeroCross_lTime[h+1]-copy_listAux_zeroCross_lTime[h])
#print(listAux_sumProcess)
mean_listAux_sumProcess = statistics.mean(listAux_sumProcess)
#print("Average: " + str(mean_listAux_sumProcess))

mean_double_listAux_sumProcess = mean_listAux_sumProcess*2
print("Two times average: " + str(mean_double_listAux_sumProcess))
freqBeat_value = 1/mean_double_listAux_sumProcess
#print(freqBeat_value)
freqBeat_FinalValue = freqBeat_value*1000
#print("The frequency is: " + str(freqBeat_FinalValue))
#Distance
R = (speed_ligth*Period_mod*freqBeat_FinalValue)/(2*2*deltaFreq)
#print("The distance is :" + str(R))
list_frequencies_ZC.append(freqBeat_FinalValue)
print("----END ZERO CROSSING-DISTANCE----")

#########################################################################
print("----START ZERO CROSSING - 2 new method - DISTANCE----")
value_len = len(listAux_zeroCross_lTime)
#print(value_len)
time = listAux_zeroCross_lTime[value_len-1] - listAux_zeroCross_lTime[0]
#print(time)
freq_NewMethod = 0.5*((value_len)-1)/(time*1e-3)
#print(freq_NewMethod)
real_freq_NewMethod = freq_NewMethod/2
print(real_freq_NewMethod)
R_ZC = (speed_ligth*Period_mod*real_freq_NewMethod)/(2*2*deltaFreq)
#print("Distance in method ZC is: " + str(R_ZC))
list_frequencies_ZC.append(real_freq_NewMethod)
#print(list_frequencies_ZC) #Hacer media a esta lista.
mean_list_frequencies_ZC = statistics.mean(list_frequencies_ZC)
#print(mean_list_frequencies_ZC)
R_mean_ZC = (speed_ligth*Period_mod*mean_list_frequencies_ZC)/(2*2*deltaFreq)
#print("Average distance: " + str(R_mean_ZC))

print("END ZERO CROSSING - 2 new method - DISTANCE----")
print("----End Distance Calculation----")
#########################################################################
print("------FFT------")
#FFT
sampling_Frequency = samples/T_halfPeriod
Y_k = np.fft.fft(subtract_list)[0:int(samples/2)]/samples
#lChannelB_aux
#subtract_list
Y_k[1:] = 2*Y_k[1:]
Pxx = np.abs(Y_k)
print(len(Pxx))
f_axis = sampling_Frequency*np.arange((samples/2))/samples
f_axis_list = []
f_axis_list = f_axis.tolist()
print(f_axis_list)
f_axis_list.pop(76)
print(len(f_axis_list))

#Find peaks in FFT.
peaksFFT_halfPeriod, _ = find_peaks(Pxx, height = 3.5)
peaksFFT_halfPeriod_list = []
peaksFFT_halfPeriod_list = peaksFFT_halfPeriod.tolist()
print(peaksFFT_halfPeriod_list)

peaksFFT_list_Pxx = []
for l in peaksFFT_halfPeriod_list:
    peaksFFT_list_Pxx.append(Pxx[l])
print(peaksFFT_list_Pxx)

peaksFFT_list_f_axis = []
for f in peaksFFT_halfPeriod_list:
    peaksFFT_list_f_axis.append(f_axis_list[f])
print(peaksFFT_list_f_axis)

value_sum_freq = sum(peaksFFT_list_f_axis)
print(value_sum_freq)
try:
    value_Freq_average = value_sum_freq / len(peaksFFT_list_f_axis)
except ZeroDivisionError:
    value_Freq_average = 0
print("The frequency is: " + str(value_Freq_average))
#########################################################################
print("-----FFT-------")
#Sinusoid approximation (Curve fitting).
def sinfunc(t, A, w, p, c):
    return A*np.sin(w*t + p) + c

tt = np.array(lTime_aux)
yy = np.array(subtract_list)
ff = np.fft.fftfreq(len(tt), abs(tt[1]-tt[0]))
Fyy = abs(np.fft.fft(yy))
guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
guess_amp = np.std(yy) * 2.**0.5
guess_offset = np.mean(yy)
guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=21800)
#print(popt)
A, w, p, c = popt
f = w/(2.*np.pi) #Frequencia angular.
fitfunc = lambda t: A * np.sin(w*t + p) + c
#print(f)
pe = 1/f
#print(pe)
frequencia = 1/pe
#print(frequencia)
Period_real_CV = 2*pe
#print(Period_real_CV)
freq_CV = 1/Period_real_CV
#print(freq_CV)
freq_CV_Real = freq_CV*1000
#print(freq_CV_Real) #Real Frequency.
R_CV = ((speed_ligth*(1/20)*freq_CV_Real))/(2*deltaFreq)
#print("Distance in CV method is: " + str(R_CV))

#Representation ChannelA, ChannelB, halfPeriod ChannelB and halfPeriod ChannelA.
plt.figure()

plt.plot(listAuxTimeReal, listAuxChannelAReal, label="ChannelA")
plt.plot(listAuxTimeReal, listAuxChannelBReal, label="ChannelB")
#plt.plot(lTime_aux, lChannelA_aux, label="Channel A. Half period")
#plt.plot(lTime_aux, lChannelB_aux, label ="Channel B. Half period.")
#plt.plot(lTime_aux, list_agrupation_listModel_poly, label = "Poly")

plt.plot(lTime_aux, subtract_list, label = "Corrected")
#plt.plot(list_period_Time[0], list_period_ChannelB[0])
#plt.plot(list_period_Time[0], list_period_ChannelA[0])
#plt.plot(list_period_Time[0], subtract_list_full[0])
#plt.plot(list_period_Time[1], subtract_list_full[1])
#plt.plot(list_period_Time[2], subtract_list_full[2])


#plt.plot(listAuxTimeReal, listAuxChannelB_Real_corr100, label = "ChannelB *100")

#plt.plot(listAuxTimeReal, subtract_list_comp, label = "Completed corrected signal")
#plt.plot(listAuxTimeReal,list_agrupation_listModel_poly_comp,label="poly")
#plt.plot(polyline_comp, modelo_comp(polyline_comp),label="Complete")
plt.plot(polyline, modelo(polyline))

#plt.plot(tt, sinfunc(tt, *popt), label = "Sine approximation")
#plt.plot(x_axisP,y_axisP,"*k")
#plt.plot(x_axisR, y_axisR, "*k")
#plt.plot(peaks_list_lTime_B, peaks_list_channelB, "*k")
#plt.plot(peaks_l_Time_Half, peaks_l_CS_Half, "*k")

plt.plot(peaks_list_lTime, peaks_list_lChannelA, "*k")
plt.legend() 
plt.grid()
plt.show()

#FFT Representation.
plt.plot(f_axis_list, Pxx, label="FFT")
plt.plot(peaksFFT_list_f_axis, peaksFFT_list_Pxx, "*k", label="Peaks FFT")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude') 
plt.legend()
plt.grid()
plt.show()

"""fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Voltage (V)', color = color)
ax1.plot(lTime_aux, lChannelA_aux, color = color, label = " Half period - Channel A")
ax1.tick_params(axis = 'y', labelcolor = color)
ax1.legend(loc = "best")

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Voltage (mV)', color = color)
ax2.plot(lTime_aux, lChannelB_aux, color = color, label = "Half period - Channel B")
ax2.plot(lTime_aux, subtract_list, color = 'purple', label = "Corrected signal - Channel B")
#ax2.plot(tt, sinfunc(tt, *popt), color = 'red', label = "Sine curve fitting")
ax2.plot(polyline, modelo(polyline), color = 'green')
ax2.tick_params(axis = 'y', labelcolor = color)
ax2.legend(loc = "best")

plt.tight_layout()
plt.grid()
#plt.ion()
plt.show()"""


