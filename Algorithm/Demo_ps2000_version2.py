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

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

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
        self.button2=QPushButton("Simulation")
        self.button4 = QPushButton("End process")
        
        #Create layout and add widgets.
        layout=QVBoxLayout()
        layout.addWidget(self.frequency)
        layout.addWidget(self.amplitude)
        layout.addWidget(self.typeSignal)
        #layout.addWidget(self.waveformDuration)
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
        except ValueError:
            print("In the panel 'Insert Frequency' and 'Insert Amplitude' you must introduce a float type value. Please, you must use the dot ( . ), instead coma ( , ).")
            os._exit(1)
        #return (valor_frequency, valorAmplitude, tipoSenal, wave_Duration)
        return (valor_frequency, valorAmplitude, tipoSenal)

    def SignalGenerator(self):
        #valorRealFrequency, valorRealAmplitude, tipoSenalReal, value_Waveform_Duration = self.EnterData()
        valorRealFrequency, valorRealAmplitude, tipoSenalReal = self.EnterData()

        print(__doc__)

        print("Attempting to open Picoscope 2000")

        ps = ps2000.PS2000()

        print(ps.getAllUnitInfo())

        #--------------------------------------------------------------------------------------------------
        waveform_Desired_Duration = 0.01875
        #--------------------------------------------------------------------------------------------------
        #waveform_Desired_Duration = 1/valorRealFrequency
        #print(waveform_Desired_Duration)

        #Original frame:
        #obs_duration = waveform_Desired_Duration
        obs_duration = waveform_Desired_Duration
        print("Obs duration: " + str(obs_duration))
        sampling_interval = obs_duration / 4096
        print("Sampling interval: " + str(sampling_interval))

        #Sampling Interval
        (actualSamplingInterval, numberSamples, maxSamples) = ps.setSamplingInterval(sampling_interval, obs_duration)
        #Show sampling interval data.
        print("Sampling interval = %f ns" % (actualSamplingInterval * 1E9)) #¿Why 1E9?
        print("Taking samples = %d" % numberSamples)
        print("Maximum samples = %d" % maxSamples)

        #Original data: -> 5.0 = 2.0 ; probeAttenuation=1
        channelRange = ps.setChannel('A', 'DC', 2.0, 0.0, enabled=True, BWLimited=False)
        print("Chosen channel range = %d" % channelRange)

        #Original version: -> ps.setSimpleTrigger('A', 1.0, 'Falling', timeout_ms = 100, enabled=True)
        ps.setSimpleTrigger('A', 1.0, 'Rising', timeout_ms = 100, enabled=True)

        #Signal Generator Function.
        ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude, waveType = tipoSenalReal, frequency = valorRealFrequency)

        ps.runBlock()
        ps.waitReady()

        print("Waiting for awg to settle.")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger.")

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
        ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude + Diferencia, waveType = tipoSenalReal, frequency = valorRealFrequency)
                
        ps.runBlock()
        ps.waitReady()

        print("Waiting for awg to settle. (Second time).")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger. (Second time).")
        
        dataA_2 = ps.getDataV('A', numberSamples, returnOverflow=False) #Data with correct amplitude
        print(len(dataA_2))
        print(dataA_2)
        #-------------------------------------------------------------------------------
        #Amplitude correction. Beta version. 
        maximo2 = max(dataA_2)
        minimo2 = min(dataA_2)
        print(maximo2)
        print(minimo2)
        valor_picoPico2 = maximo2+(abs(minimo2))
        print("Peak to peak value 2: " + str(valor_picoPico2))
        Diferencia2 = valorRealAmplitude-valor_picoPico2
        print("Difference value 2: " + str(Diferencia2))
        amplitudReal2 = valor_picoPico2 + Diferencia2
        print("Real amplitude 2: " + str(amplitudReal2))
        #End amplitude correction. Beta version.


        """#New generation (Amplitude corrected)
        ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude + Diferencia + Diferencia2, waveType = tipoSenalReal, frequency = valorRealFrequency)
                
        ps.runBlock()
        ps.waitReady()

        print("Waiting for awg to settle. (Second time).")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger. (Second time).")
        
        dataA_3 = ps.getDataV('A', numberSamples, returnOverflow=False) #Data with correct amplitude
        print(len(dataA_3))

        #------------------------------------------------------------------------------------------------
        #Amplitude correction. Beta version. 
        maximo3 = max(dataA_3)
        minimo3 = min(dataA_3)
        valor_picoPico3 = maximo3+(abs(minimo3))
        print("Peak to peak value 3: " + str(valor_picoPico3))
        Diferencia3 = valorRealAmplitude-valor_picoPico3
        print("Difference value 3: " + str(Diferencia3))
        amplitudReal3 = valor_picoPico3 + Diferencia3
        print("Real amplitude 3: " + str(amplitudReal3))
        #End amplitude correction. Beta version.


        #New generation (Amplitude corrected)
        ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude + Diferencia + Diferencia2 + Diferencia3, waveType = tipoSenalReal, frequency = valorRealFrequency)
                
        ps.runBlock()
        ps.waitReady()

        print("Waiting for awg to settle. (Third time).")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger. (Third time).")
        
        dataA_4 = ps.getDataV('A', numberSamples, returnOverflow=False) #Data with correct amplitude
        print(len(dataA_4))

        #------------------------------------------------------------------------------------------------
        #Amplitude correction. Beta version. 
        maximo4 = max(dataA_4)
        minimo4 = min(dataA_4)
        valor_picoPico4 = maximo4+(abs(minimo4))
        print("Peak to peak value 4: " + str(valor_picoPico4))
        Diferencia4 = valorRealAmplitude-valor_picoPico4
        print("Difference value 4: " + str(Diferencia4))
        amplitudReal4 = valor_picoPico4 + Diferencia4
        print("Real amplitude 4: " + str(amplitudReal4))
        #End amplitude correction. Beta version.


        #New generation (Amplitude corrected)
        ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude + Diferencia + Diferencia2 + Diferencia3 + Diferencia4, waveType = tipoSenalReal, frequency = valorRealFrequency)
                
        ps.runBlock()
        ps.waitReady()

        print("Waiting for awg to settle. (Fourd time).")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger. (Fourd time).")
        
        dataA_5 = ps.getDataV('A', numberSamples, returnOverflow=False) #Data with correct amplitude
        print(len(dataA_5))


         #------------------------------------------------------------------------------------------------
        #Amplitude correction. Beta version. 
        maximo5 = max(dataA_5)
        minimo5 = min(dataA_5)
        valor_picoPico5 = maximo5+(abs(minimo5))
        print("Peak to peak value 5: " + str(valor_picoPico5))
        Diferencia5 = valorRealAmplitude-valor_picoPico5
        print("Difference value 5: " + str(Diferencia5))
        amplitudReal5 = valor_picoPico5 + Diferencia5
        print("Real amplitude 5: " + str(amplitudReal5))
        #End amplitude correction. Beta version.


        #New generation (Amplitude corrected)
        #ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude + Diferencia + Diferencia2 + Diferencia3 + Diferencia4 + Diferencia 5, waveType = tipoSenalReal, frequency = valorRealFrequency)
        ps.setSigGenBuiltInSimple(offsetVoltage = 0, pkToPk = valorRealAmplitude + Diferencia5, waveType = tipoSenalReal, frequency = valorRealFrequency)
                
        ps.runBlock()
        ps.waitReady()

        print("Waiting for awg to settle. (Five time).")
        time.sleep(2.0)
        ps.runBlock()
        ps.waitReady()
        print("Done waiting for trigger. (Five time).")
        
        dataA_6 = ps.getDataV('A', numberSamples, returnOverflow=False) #Data with correct amplitude
        print(len(dataA_6))"""

        

        #-------------------------------------------------------------------------------
        # Y-axis calculation. 
        dataTimeAxis = np.arange(numberSamples) * actualSamplingInterval
        
        print("Data time axis: " + str(dataTimeAxis))
        print("Data time axis length: " + str(len(dataTimeAxis)))
        #-------------------------------------------------------------------------------
        #Conversion de datos.

        ps.stop()
        ps.close()

        return (dataTimeAxis, dataA, dataA_2, actualSamplingInterval, numberSamples, waveform_Desired_Duration)

    #Module end script.
    def stopProcess(self):
        sys.exit()

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

    def cyclesPreparation1seg(self):
        #Prepararlo para un segundo. Hacemos calculo para un segundo y luego hacemos bucle loop para que se introduzca la medida que quiere el
        #investigador realizar y multiplicamos el ciclo.

        #Data preparation
        timeAxis, originalData, correctData, ActualSamplingInterval, samples, waveformDuration = self.SignalGenerator()

        print(type(timeAxis))
        print(type(correctData))
        VM = correctData.tolist()
        #VM_2 = correctData_2.tolist()
        TA = timeAxis.tolist()
        print(type(VM))
        #print(type(VM_2))
        print(type(TA))
        
        #vecesCiclo = 1
        #estructureTime = vecesCiclo*waveformDuration
        estructureTime = waveformDuration
        print(estructureTime)
        """muestras = 3750
        waveformDuration = 1200e-6
        samplingInterval = waveformDuration / muestras"""

        #Para alcanzar el 1.2 seg.
        #correctDataCompleted = vecesCiclo * VM
        #correctDataCompleted2 = vecesCiclo * VM_2

        correctDataCompleted = VM
        #correctDataCompleted2 = VM_2
        
        print(len(correctDataCompleted))
        #-------------------------------------------------------------------------------

        return(correctDataCompleted, VM, TA, ActualSamplingInterval, estructureTime)

        """Periodo = 1/valorRealFrequency
        print("Periodo: " + str(Periodo))
        FrequenciaMuestreo = 2*valorRealFrequency
        print("Frequencia de muestreo: " + str(FrequenciaMuestreo))
        periodoMuestreo = 1/FrequenciaMuestreo
        print("Periodo de muestreo: " + str(periodoMuestreo))"""

    #Signal Representation, Excel creation and Telegram Bot. 
    def SignalGeneratorRepresentation(self):

        #tiempoAxis, valor_Original, valor_modificado = self.SignalGenerator()

        Datacorrecto, valor_modificado_DataCorrecto, tiempoAxis, actual_Sampling_Interval, Estructure_Time = self.cyclesPreparation1seg()
        print(type(Datacorrecto))
        print(len(Datacorrecto))

        print(actual_Sampling_Interval)
        
        tiempoOriginal = []
        #tiempoOriginal = self.drange(0, 2.4e-3, actual_Sampling_Interval, 14)
        tiempoOriginal = self.drange(0, Estructure_Time, actual_Sampling_Interval, 13)
        print(tiempoOriginal)
        print(type(tiempoOriginal))
        print(len(tiempoOriginal))

        #tiempoOriginal.remove(2.4e-3)
        #tiempoOriginal.remove(Estructure_Time)
        tiempoOriginal.pop(3126)
        print(len(tiempoOriginal))
        
        
        """print(type(tiempoAxis))
        print(type(valor_modificado))
        VM = valor_modificado.tolist()
        TA = tiempoAxis.tolist()

        print(VM)
        print(TA)
        print(type(VM))
        print(type(TA))

        print(len(VM))
        print(len(TA))

        print(valor_modificado)
        print(tiempoAxis)"""

        #----------------------------------------------------------------------------------------------------
        #Excel creation.
        """df = pd.DataFrame({'V data': [valor_modificado],
                           'Time data': [tiempoAxis]})

        df = df[['V data', 'Time data']]

        writer = ExcelWriter('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/excel_Folder/datas.xlsx')
        df.to_excel(writer, 'Data sheet')
        writer.save()"""        
        #----------------------------------------------------------------------------------------------------
        #Signal representation. 
        plt.figure()
        
        #plt.plot(tiempoAxis, valor_modificado, label="Clock")
        #plt.plot(tiempoAxis, valor_Original, label="Clock")
        plt.plot(tiempoOriginal, Datacorrecto, label = "Clock")
        #plt.plot(nuealista, Datacorrecto2, label = "Clock")
        #plt.plot(tiempoAxis, Datacorrecto2, label = "Clock")

        plt.ion()
        plt.grid(True, which='major')
        plt.title("Picoscope oscilloscope.")
        plt.ylabel("Voltage (V).")
        plt.xlabel("Time (s).")
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


