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

from picoscope import ps2000
from time import sleep 

from PySide6.QtWidgets import (QLineEdit, QPushButton, QApplication, QVBoxLayout, QDialog, QLabel, QMainWindow)
from os import remove
from os import path
from pynput import keyboard as kb

from pandas import ExcelWriter

#Telegram's Bot.
TOKEN = '5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4'
chat_id = '9284473'
bot =telebot.TeleBot(TOKEN)
bot.config['api_key'] = TOKEN
#Bot status.
print(bot.get_me())

#----------------------------------------------------------------------------------------
#Text box. Explains how to enter data.
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
        
#Text box end code.
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
#Menu for data input: Frequency, Amplitude, Type of signal.
class Form(QDialog):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        #Create widgets.
        self.frequency = QLineEdit("Insert frequency: ")
        self.amplitude = QLineEdit("Insert amplitude: ")
        self.typeSignal = QLineEdit("Insert type of signal: ")
        #self.waveformDuration = QLineEdit("Insert waveform duration: ")
        self.voltageOffset = QLineEdit("Insert voltage offset: ")
        self.instantsMeasure = QLineEdit("Insert number of instants to measure: ")
        
        self.button2=QPushButton( "Simulation")
        self.button4 = QPushButton( "End process")
        
        #Create layout and add widgets.
        layout=QVBoxLayout()
        layout.addWidget(self.frequency)
        layout.addWidget(self.amplitude)
        layout.addWidget(self.typeSignal)
        #layout.addWidget(self.waveformDuration)
        layout.addWidget(self.voltageOffset)
        layout.addWidget(self.instantsMeasure)
        layout.addWidget(self.button2)
        layout.addWidget(self.button4)
        
        #Set dialog layout.
        self.setLayout(layout)
        
        #Add button signal to greetings slot.
        self.button2.clicked.connect(self.SignalGeneratorRepresentation)
        self.button4.clicked.connect(self.stopProcess)
        
    def EnterData(self):
        #Collection and allocation of information collected by the interface.
        try:
            #Frequency
            valor = float(self.frequency.text())
            valor_frequency = valor
            #Amplitude
            valorA = float(self.amplitude.text())
            valorAmplitude = valorA
            #Type signal
            typeSignal_D = str(self.typeSignal.text())
            tipoSenal = typeSignal_D
            """#Waveform Duration
            waveform_Duration = float(self.waveformDuration.text())
            wave_Duration = waveform_Duration"""
            #Type voltage offset.
            vOffset = float(self.voltageOffset.text())
            value_vOffset = vOffset
            #Type number instant.
            numInstant = float(self.instantsMeasure.text())
            value_numInstant = numInstant
        except ValueError:
            print("In the panel 'Insert Frequency' and 'Insert Amplitude' you must introduce a float type value. Please, you must use the dot ( . ), instead coma ( , ).")
            os._exit(1)
        #return (valor_frequency, valorAmplitude, tipoSenal, wave_Duration)
        return (valor_frequency, valorAmplitude, tipoSenal, value_vOffset, value_numInstant)

    def SignalGenerator(self):
        #valorRealFrequency, valorRealAmplitude, tipoSenalReal, value_Waveform_Duration = self.EnterData()
        valorRealFrequency, valorRealAmplitude, tipoSenalReal, value_Voffset, value_numInstant = self.EnterData()

        print(__doc__)

        print("Attempting to open Picoscope 2000")

        ps = ps2000.PS2000()

        print(ps.getAllUnitInfo())
        #--------------------------------------------------------------------------------------------------
        #waveform_Desired_Duration = 9600e-6
        waveform_Desired_Duration = 0.6144
        #--------------------------------------------------------------------------------------------------
        #Original frame:
        obs_duration = waveform_Desired_Duration
        print("Obs duration: " + str(obs_duration))
        sampling_interval = obs_duration / 4096
        print("Sampling interval: " + str(sampling_interval))

        #Sampling Interval
        (actualSamplingInterval, numberSamples, maxSamples) = ps.setSamplingInterval(sampling_interval, obs_duration)
        print(actualSamplingInterval)
        #Show sampling interval data.
        print("Sampling interval = %f ns" % (actualSamplingInterval * 1E6)) #¿Why 1E9?
        print("Sampling interval = " + str(actualSamplingInterval))
        
        print("Taking samples = %d" % numberSamples)
        print("Maximum samples = %d" % maxSamples)

        channelRange = ps.setChannel('A', 'DC', 20.0, 5.0, enabled=True, BWLimited=False, probeAttenuation=10.0)
        #print("Chosen channel range = %d" % channelRange)

        channelRange2 = ps.setChannel('B', 'DC', 50e-3, 0.0, enabled=True, BWLimited=False, probeAttenuation=1) #ChannelB
        #print("Chosen channel range = %d" % channelRange2)

        #Original version: -> ps.setSimpleTrigger('A', 1.0, 'Falling', timeout_ms = 100, enabled=True)
        ps.setSimpleTrigger('A', 1.0, 'Rising', timeout_ms = 100, enabled=True)

        #Signal Generator Function.
        ps.setSigGenBuiltInSimple(offsetVoltage = value_Voffset, pkToPk = valorRealAmplitude, waveType = tipoSenalReal, frequency = valorRealFrequency)

        ps.runBlock()
        ps.waitReady()

        """print("Waiting for awg to settle.")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger.")"""

        dataA = ps.getDataV('A', numberSamples, returnOverflow=False) #Original data with lossess.
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
        #End amplitude correction. Beta version. 
        #-------------------------------------------------------------------------------
        #New generation (Amplitude corrected)
        ps.setSigGenBuiltInSimple(offsetVoltage = value_Voffset, pkToPk = valorRealAmplitude + Diferencia, waveType = tipoSenalReal, frequency = valorRealFrequency)
                
        ps.runBlock()
        ps.waitReady()

        """print("Waiting for awg to settle. (Second time).")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger. (Second time).")"""
        
        dataA_2 = ps.getDataV('A', numberSamples, returnOverflow=False) #Data with correct amplitude
        print(len(dataA_2))
        print(dataA_2)
        #-------------------------------------------------------------------------------
        # Y-axis calculation. 
        dataTimeAxis = np.arange(numberSamples) * actualSamplingInterval
        print("Data time axis length: " + str(len(dataTimeAxis)))
        print(dataTimeAxis)
        #-------------------------------------------------------------------------------
        ps.stop()
        ps.close()
        #dataA: --> Dato original. dataA_2 --> Dato original corregido.
        return (dataTimeAxis, dataA, dataA_2, actualSamplingInterval, numberSamples, waveform_Desired_Duration)

    #Data time axis Module. (Full cycle).
    def drange(self, start, end, increment, round_decimal_places=None):
        result = []
        if start < end:
            #Counting up, e.g. 0 to 0.4 in 0.1 increments.
            #if increment < 0:
                #raise Exception("Error: When counting up, increment must be positive.")
            while start <= end:
                result.append(start)
                start += increment
                if round_decimal_places is not None:
                    start = round(start, round_decimal_places)
        return result

    def cyclesPreparation(self):
        #Data preparation
        timeAxis, originalData, correctData, Actual_Sampling_Interval, samples, waveformDuration = self.SignalGenerator()
        freq, amp, senal = self.EnterData()
        #-------------------------------------------------------------------------------
        peri = 1/freq
        print("Periodo: " + str(peri))
        #-------------------------------------------------------------------------------
        #Conversion de listas.
        print(type(timeAxis))
        print(type(correctData))
        print(len(timeAxis))
        print(len(correctData))
        #Conversion a lista.
        VM = correctData.tolist() #Valor de voltaje
        TA = timeAxis.tolist() #Tiempo
        print(type(VM))
        print(type(TA))
        print(len(VM))
        print(len(TA))
        #Visualizacion de valores.
        print(VM)
        print(TA)

        #Procesamiento de Valores de Amplitud.
        muestrasCor=int(peri/Actual_Sampling_Interval) #Muestras para representar un periodo.
        print("Muestras de 1 periodo: " + str(muestrasCor+1))
        listaAmplitudM = [] #Lista auxiliar amplitud.
        listaAmplitudM = VM.copy() #Copia de lista CorrecData.
        print(len(listaAmplitudM))
        del listaAmplitudM[muestrasCor+1:len(VM)] #Borrar indices a partir de muestrasCor + 2 de la lista.
        print(len(listaAmplitudM))
        print(listaAmplitudM)

        #Procesamiento de Valores de Tiempo.
        listaTiempoM = [] #Lista auxiliar tiempo.
        listaTiempoM = TA.copy() #Copia de lista timeAxis
        print(len(listaTiempoM))
        del listaTiempoM[muestrasCor+1:len(TA)] #Borrar indices a partir de muestrasCor + 2 de la lista.
        print(len(listaTiempoM))
        print(listaTiempoM)
        #-------------------------------------------------------------------------------
        # Multiplicar por el numero de ciclo a elegir.
        #vecesCiclo = 1
        #estructureTime = vecesCiclo*waveformDuration
        estructureTime = waveformDuration
        print("Waveform duration: " +str(estructureTime))
        print(len(VM))
        print(len(TA))
        #-------------------------------------------------------------------------------
        return(VM, TA, Actual_Sampling_Interval, estructureTime, listaAmplitudM, listaTiempoM) #VM = CorrectDataCompleted.

    #Modules correspond to cycles Implementation
    def cyclesImplementationFull(self):
        valor_modificado_DataCorrecto, tiempoAxis, actual_Sampling_Interval, Estructure_Time, lisA, lisT = self.cyclesPreparation()

        print(type(valor_modificado_DataCorrecto))
        print(len(valor_modificado_DataCorrecto))
        print("Actual sampling interval: " +str(actual_Sampling_Interval))
        print(len(tiempoAxis))
        #-------------------------------------------------------------------------------
        print(len(lisA))
        print(len(lisT))

        ciclo = 60
        TiempoTotal = ciclo*Estructure_Time
        tiempoTotalModificado = self.drange(0,TiempoTotal-actual_Sampling_Interval,actual_Sampling_Interval, 12)

        print(type(tiempoTotalModificado))
        print(len(tiempoTotalModificado))
        print(tiempoTotalModificado)
    
        amplitudListaModificado = ciclo*valor_modificado_DataCorrecto
        print(len(amplitudListaModificado))

        return (tiempoTotalModificado, amplitudListaModificado, lisT, lisA, tiempoAxis, valor_modificado_DataCorrecto)


    #Signal Representation, Excel creation and Telegram Bot. 
    def SignalGeneratorRepresentation(self):
        #valor_modificado_DataCorrecto, tiempoAxis, actual_Sampling_Interval, Estructure_Time, lisA, lisT = self.cyclesPreparation()

        tiempoTotalCiclo, AmplitudCiclo, lisTperiod, lisAperiod, tiempo_Axis, Amplitud_modified_CorrectData = self.cyclesImplementationFull()
        
        freq1, amp1, senal1 = self.EnterData()
        print("La amplitud es: " + str(amp1))
        
        #Excel creation.
        """df = pd.DataFrame({'V data': [valor_modificado],
                           'Time data': [tiempoAxis]})

        df = df[['V data', 'Time data']]

        writer = ExcelWriter('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/excel_Folder/datas.xlsx')
        df.to_excel(writer, 'Data sheet')
        writer.save()"""        
        #-------------------------------------------------------------------------------
        #Signal representation. 
        plt.figure()
        
        #plt.plot(tiempoAxis, valor_modificado, label="Clock")
        #plt.plot(tiempoAxis, valor_Original, label="Clock")
        #plt.plot(tiempoOriginal, Datacorrecto, label = "Clock")

        plt.plot(tiempo_Axis, Amplitud_modified_CorrectData, label="Clock") #Original

        plt.plot(lisTperiod, lisAperiod, label="Clock") #Representacion del periodo de la onda.
        
        #plt.plot(tiempoTotalCiclo, AmplitudCiclo, label="Clock") #Representación del ciclo.


        plt.ion()
        plt.grid(True, which='major')
        plt.title("Picoscope oscilloscope.")
        plt.ylabel("Voltage (V).")
        plt.xlabel("Time (s).")
        plt.legend()
        
        """#Message to Telegram Bot. Information about the data.
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

    #Module end script.
    def stopProcess(self):
        sys.exit()

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


