from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import time
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import tkinter as tk
import os, sys
import json
import datetime
import csv
import pandas as pd
import telebot
import requests
import time

from picoscope import ps2000
from time import sleep 

from PySide6.QtWidgets import (QLineEdit, QPushButton, QApplication, QVBoxLayout, QDialog, QLabel, QMainWindow)
from os import remove
from os import path
from pynput import keyboard as kb

#Bot de Telegram.
TOKEN = '5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4'
chat_id = '9284473'
bot =telebot.TeleBot(TOKEN)
bot.config['api_key'] = TOKEN
#Estado del bot.
print(bot.get_me())

#----------------------------------------------------------------------------------------
#Cuadro de texto sobre como introducir datos. 
class Ayuda_Dialog:
    def __init__(self, parent):
        text = ("Information to be taken into account:\n"
                "\n"
                "To exit the box help and be able to enter the data, you must click \n"
                "the tab 'X'. \n"
                "\n"
                "Different steps: \n"
                "Step 1: Insert the value of the frequency. Use a dot ( . ), instead of a \n"
                "coma ( , ).\n" 
                "Step 2: Insert the value of the amplitude. Use a dot ( . ), instead of a \n"
                "coma ( , ).\n"
                "Step 3: Insert the type of the signal. The different types are: Sine, Square, \n"
                "Triangle, RampUp, RampDown and DCVoltage.\n"
                "\n"
                "To introduce values, we need to select the panel and erase the quote. After \n"
                "that, we can introduce the value of the frequency, amplitude and type \n"
                "of signal. Once we have entered the data, we press the button 'Results' \n"
                "and we can see the results obtained. \n"
                "\n"
                "If we want to change the value, we just need to modified the values and \n"
                "press again the buttom. If we change values, the results will be different \n"
                "Finally, if we don't press the button, the image will dissapear in the \n"
                "following 60 seconds. \n")

        self.top = tk.Toplevel(parent)
        self.top.title("Help")
        display = tk.Text(self.top)
        display.pack()
        display.insert(tk.INSERT, text)
        display.config(state=tk.DISABLED)
        b = tk.Button(self.top, text="Close", command=self.cerrar)
        b.pack(pady=5)

    def cerrar(self):
        self.top.destroy()

class Main_Window:
    def __init__(self,  root):
        root.geometry("200x100")
        tk.Button(root, text="Information",  command = self.ayuda).pack()
    def ayuda(self):
        Ayuda_Dialog(root)
#Fin de codigo parte: Cuadro de texto sobre como introducir datos.
#---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
#Menu para introducir datos: Frequency, Amplitude, Type of signal.
class Form(QDialog):

    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        
        #Create widgets.
        #self.frequency = QLineEdit("Insert frequency: ")
        #self.amplitude = QLineEdit("Insert amplitude: ")
        #self.typeSignal = QLineEdit("Insert type of signal: ")
        #self.waveformDuration = QLineEdit("Insert waveform duration: ")
        self.button2=QPushButton("Simulation")
        self.button4 = QPushButton("End process")
        
        #Create layout and add widgets.
        layout=QVBoxLayout()
        #layout.addWidget(self.frequency)
        #layout.addWidget(self.amplitude)
        #layout.addWidget(self.typeSignal)
        #layout.addWidget(self.waveformDuration)
        layout.addWidget(self.button2)
        layout.addWidget(self.button4)
        
        #Set dialog layout.
        self.setLayout(layout)
        
        #Add button signal to greetings slot.
        self.button2.clicked.connect(self.SignalGeneratorRepresentation)
        self.button4.clicked.connect(self.stopProcess)
        
    def SignalGenerator(self):
        #Recogida y asignacion de la informacion recogida por la interfaz.
        #try:
            #Frequency
            #valor = float(self.frequency.text())
            #valor_frequency = valor
            #Amplitude
            #valorA = float(self.amplitude.text())
            #valorAmplitude = valorA
            #Type signal
            #typeSignal_D = str(self.typeSignal.text())
            #tipoSenal = typeSignal_D
            #Waveform Duration
            #waveform_Duration = float(self.waveformDuration.text())
            #wave_Duration = waveform_Duration
        #except ValueError:
            #print("In the panel 'Insert Frequency' and 'Insert Amplitude' you must introduce a float type value. Please, you must use the dot ( . ), instead coma ( , ).")
            #os._exit(1)

        #Correspondencia de valores introducidos a nuevas variables. 
        valorRealFrequency = 50000
        valorRealAmplitude = 4
        tipoSenalReal = "Triangle"
        #value_Waveform_Duration = wave_Duration

        #valorRealFrequency = valor_frequency
        #valorRealAmplitude = valorAmplitude
        #tipoSenalReal = tipoSenal

        print(__doc__)

        print("Attempting to open Picoscope 2000")

        ps = ps2000.PS2000()

        print(ps.getAllUnitInfo())

        waveform_Desired_Duration = 50E-6 #Original
        #waveform_Desired_Duration = value_Waveform_Duration
        
        #Original: *3.
        obs_duration = 3*waveform_Desired_Duration
        print("Obs duration: " + str(obs_duration))
        
        sampling_interval = obs_duration / 4096
        print("Sampling interval: " + str(sampling_interval))
        
        (actualSamplingInterval, numberSamples, maxSamples) = ps.setSamplingInterval(sampling_interval, obs_duration)
        print(actualSamplingInterval)

        #print("Sampling interval sin modificar: " % actualSamplingInterval)
        print("Sampling interval = %f ns" % (actualSamplingInterval * 1E9)) #¿Porque 1E9?
        print("Taking samples = %d" % numberSamples)
        print("Maximum samples = %d" % maxSamples)

        channelRange = ps.setChannel('A', 'DC', 2.0, 0.0, enabled=True, BWLimited=False, probeAttenuation=1)
        print("Chosen channel range = %d" % channelRange)

        ps.setSimpleTrigger('A', 1.0, 'Falling', timeout_ms = 100, enabled=True)

        pico1 = ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude, waveType = tipoSenalReal, frequency = valorRealFrequency)
        #pico1 = ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = 1.2, waveType = "Triangle", frequency = 1000)
        print(pico1)

        #pico_awg = ps.setAWGSimple(waveform = (ystuple), du

        ps.runBlock()
        ps.waitReady()

        print("Waiting for awg to settle.")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger")

        dataA = ps.getDataV('A', numberSamples, returnOverflow=False) #Datos originales (con perdidas)

        #-------------------------------------------------------------------------------
        #Amplitude correction. Beta version. 
        maximo = max(dataA)
        minimo = min(dataA)
        valor_picoPico = maximo+(abs(minimo))
        print("Peak to peak value: " + str(valor_picoPico))
        Diferencia = valorRealAmplitude-valor_picoPico
        print("Difference value: " + str(Diferencia))
        
        amplitudReal = valor_picoPico + Diferencia
        print("Real amplitude: " + str(amplitudReal))
        #print(maximo)
        #print(minimo)

        #New generation (Amplitude corrected)
        pico2 = ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude + Diferencia, waveType = tipoSenalReal, frequency = valorRealFrequency)
        print(pico2)
    
        ps.runBlock()
        ps.waitReady()

        print("Waiting for awg to settle.")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger")

        dataA_2 = ps.getDataV('A', numberSamples, returnOverflow=False) #Datos con amplitud corregida.

        maximo2 = max(dataA_2)
        minimo2 = min(dataA_2)
        valor_PicoPico2 = maximo2+(abs(minimo2))
        print("Valor pico a pico2: " + str(valor_PicoPico2))
        Diferencia2 = valorRealAmplitude-valor_PicoPico2
        print("Difernecia2: " +str(Diferencia2))
        amplitudReal2 = valor_PicoPico2 + Diferencia2
        print("Real amplitude2 :" + str(amplitudReal2))

        #End Amplitude Correction. Beta version.
        #-------------------------------------------------------------------------------

        """
        #-------------------------------------------------------------------------------
        #Amplitude correction. Beta version. 
        maximo = max(dataA)
        minimo = min(dataA)
        valor_picoPico = maximo+(abs(minimo))
        print("Peak to peak value: " + str(valor_picoPico))
        Diferencia = valorRealAmplitude-valor_picoPico
        print("Difference value: " + str(Diferencia))
        
        amplitudReal = valor_picoPico + Diferencia
        print("Real amplitude: " + str(amplitudReal))
        print(maximo)
        print(minimo)
        
        data_matriz = []
        data_matriz = list(dataA)
        print(max(data_matriz))
        print(min(data_matriz))
        print(data_matriz)
        print("List length: " + str(len(data_matriz)))

        data_matriz_auxiliar = []
        data_matriz_auxiliar.append(Diferencia)
        data_matriz_Real = []
        veces = (len(data_matriz))
        print(veces)
        data_matriz_Real = data_matriz_auxiliar*veces
        print(data_matriz_Real)
        print("list length (2): " + str(len(data_matriz_Real)))
        suma_lista=[]
        for i in range(len(data_matriz)):
            if data_matriz[i] < 0:
                suma_lista.append(data_matriz[i]-data_matriz_Real[i])
            else:
                suma_lista.append(data_matriz[i]+data_matriz_Real[i])
        print(suma_lista)
        print("List length (3): " + str(len(suma_lista)))
        print(max(suma_lista))
        print(min(suma_lista))"""

        dataTimeAxis = np.arange(numberSamples) * actualSamplingInterval
        print("Data time axis: " + str(dataTimeAxis))
        print("La longitud de data time axis es: " + str(len(dataTimeAxis)))

        ps.stop()
        ps.close()

        return (dataTimeAxis, dataA, dataA_2, valorRealFrequency, valorRealAmplitude, tipoSenalReal)

    #End script
    def stopProcess(self):
        sys.exit()


    def SignalGeneratorRepresentation(self):

        tiempoAxis, valor_Original, valor_modificado, frequency, amplitude, typeSignal = self.SignalGenerator()

        plt.figure()

        #Correction attempt.
        #plt.plot(dataTimeAxis, valor_Original, label="Clock")
        plt.plot(tiempoAxis, valor_modificado, label="Clock")
        #plt.plot(tiempoAxis, valor_Original, label="Clock")
        
        plt.ion()
        plt.grid(True, which='major')
        plt.title("Picoscope waveforms")
        plt.ylabel("Voltage (V)")
        plt.xlabel("Time (s)")
        plt.legend()

        """
        #Message to Telegram Bot. Information about the data.
        bot.send_message('9284473', 'Welcome to Radar Information bot. Here, you can see the obtained results.')
        #-------------------------------------------------------------------------------

        #Send txt file to telegram Bot (just if it is necessary).
        #Create txt file.
        with open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/file_Information.txt', 'w') as archivo:
            archivo.write("The value of the frequency is: " + str(frequency) + " Hz" + "\n"
                          "The value of the amplitude is: " + str(amplitude) + " V" + "\n"
                          "The type of the signal is: " + str(typeSignal) + "\n"
                          )

        #Send the document to telegram bot.
        today= datetime.datetime.now()
        files = {'document':open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/file_Information.txt','rb')}
        resp = requests.post('https://api.telegram.org/bot5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4/sendDocument?chat_id=9284473&caption=File txt'.format(today), files=files)
        
        #Send image to telegram Bot. 
        plt.savefig('picture.png') #Image
        
        files = {'photo':open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/picture.png','rb')}
        resp = requests.post('https://api.telegram.org/bot5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4/sendPhoto?chat_id=9284473&caption=graph'.format(today), files=files)
        print(resp.status_code)"""
        #-------------------------------------------------------------------------------
        
        plt.show()
    
        #plt.waitforbuttonpress(3060) #60 seg.
        #plt.close()
#---------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    #Information about enter data.
    root = tk.Tk()
    Main_Window(root)
    root.mainloop()
                                
    #Second class 'QApplication'
    app = QApplication(sys.argv)

    #Create and show the form.
    form = Form()
    form.show()

    #Run the main Qt loop.
    sys.exit(app.exec_())


