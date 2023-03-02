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

from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

file1 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_01.csv')
file2 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_02.csv')
file3 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_03.csv')
file4 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_04.csv')
file5 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_05.csv')
file6 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_06.csv')
file7 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_07.csv')
file8 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_08.csv')
file9 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_09.csv')
file10 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_10.csv')
file11 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_11.csv')
file12 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fourthTime/4 medida - modulo 1 - free space (100mV)/1m - modulo 1 - free space/1m - modulo 1 - free space_12.csv')

#Constants
freq = 20
T = 1/freq
T_halfPeriod= T/2
speed_ligth = 2.99e8
point_FreqHigh = 24.196e9
point_FreqLow = 24.082e9
deltaFreq = point_FreqHigh - point_FreqLow


def read_csv(filename):
    data = pd.read_csv(filename, header = 0, sep = ";", decimal = ",")

    time = data['Tiempo']
    time.pop(0)
    lTime = [float(x.replace(',','.')) for x in time]

    channelA = data['Canal A']
    channelA.pop(0)
    lChannelA = [float(x.replace(',', '.')) for x in channelA]

    channelB = data['Canal B']
    channelB.pop(0)
    lChannelB = [float(x.replace(',', '.')) for x in channelB]
    return (lTime, lChannelA, lChannelB)
############################################################################################

def samples_IntervalTime(lTime):
    #Frequency, samples and period.
    intervalTime = abs(lTime[10]-lTime[9])
    intervalTime_seg = intervalTime/1000

    freq_Sampling = 1/intervalTime_seg

    samples_Period = round(T / intervalTime_seg)
    #print(samples_Period)

    samples_halfPeriod = T_halfPeriod / intervalTime_seg
    samples_halfPeriod_Round = round(samples_halfPeriod)
    #print(samples_halfPeriod_Round)

    samplesRound_2half = samples_halfPeriod_Round/2
    #print(samplesRound_2half)

    samples_2half_Round = round(samplesRound_2half)+1
    #print(samples_2half_Round)
    return(samples_halfPeriod_Round, samples_Period)
############################################################################################
def processing_signal_section(lTime, lChannelA, lChannelB, samples_halfPeriod_Round, samples_Period):
    ###List processing to catch half-period signals.
    lTime_aux = []
    lTime_aux = lTime.copy()
    wave_Period = 3
    del lTime_aux[(samples_halfPeriod_Round*wave_Period):len(lTime)]
    del lTime_aux[0:samples_Period]
    #print(lTime_aux)
    #print(len(lTime_aux))

    lChannelA_aux = []
    lChannelA_aux = lChannelA.copy()
    del lChannelA_aux[(samples_halfPeriod_Round*wave_Period):len(lChannelA)]
    del lChannelA_aux[0:samples_Period]

    lChannelB_aux = []
    lChannelB_aux = lChannelB.copy()
    del lChannelB_aux[(samples_halfPeriod_Round*wave_Period):len(lChannelB)]
    del lChannelB_aux[0:samples_Period]

    return(lTime_aux, lChannelA_aux, lChannelB_aux)
############################################################################################
def polynomial_Approximation(lTime_aux, lChannelB_aux, grade):
    modelo = np.poly1d(np.polyfit(lTime_aux, lChannelB_aux, grade))
    polyline = np.linspace(lTime_aux[0], lTime_aux, len(lTime_aux))

    modelo_poly = []
    modelo_poly = modelo(polyline)
    agrupation_modelo_poly = modelo_poly[len(lTime_aux)-1]

    l_agrupation_modelo_poly = agrupation_modelo_poly.tolist()
    subtract = [t1-t2 for t1, t2 in zip(lChannelB_aux, l_agrupation_modelo_poly)]

    return(subtract, polyline, modelo)
############################################################################################



############################################################################################
def representation(lTime, lChannelA, lChannelB, lTime_aux, lChannelA_aux, lChannelB_aux, polyline, modelo, subtract):
    plt.figure()
    plt.plot(lTime, lChannelA, label = "ChannelA", color = "blue")
    plt.plot(lTime, lChannelB, label = "ChannelB", color = "green")
    plt.plot(lTime_aux, lChannelA_aux, label = "Half ChannelA", color = "purple")
    plt.plot(lTime_aux, lChannelB_aux, label = "Half ChannelB", color = "orange")
    plt.plot(lTime_aux, subtract, color = "brown", label = "Corrected")
    plt.plot(polyline, modelo(polyline), color = "red")
    plt.legend()
    plt.grid()
    plt.show()


(Time, channelA, channelB) = read_csv(file4)

(samples_half, samples_full) = samples_IntervalTime(Time)

(Time_half, channelA_half, channelB_half) = processing_signal_section(Time, channelA, channelB, samples_half, samples_full)

(subtract, polyline, modelo) = polynomial_Approximation(Time_half, channelB_half, 4)

representation(Time, channelA, channelB, Time_half, channelA_half, channelB_half, polyline, modelo, subtract)



"""fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Voltage (V)', color = color)
ax1.plot(lTime, lChannelA, color = color, label = "ChannelA")
ax1.tick_params(axis = 'y', labelcolor = color)
ax1.legend(loc = "best")

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Voltage(mV)', color = color)
ax2.plot(lTime, lChannelB, color = color, label = "ChannelB")
ax2.tick_params(axis = 'y', labelcolor = color)
ax2.legend(loc = "best")

plt.tight_layout()
plt.grid()
plt.show()"""
              

