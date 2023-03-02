import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette

"""
def show():
    result = combo.currentText()
    print(result)



def show2():
    print(line.text())

if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fussion")

    qp = QPalette()
    qp.setColor(QPalette.ButtonText, Qt.black)
    qp.setColor(QPalette.Window, Qt.black)
    qp.setColor(QPalette.Button, Qt.gray)
    app.setPalette(qp)


    w = QWidget()

    #w = QMainWindow()
    w.setGeometry(400, 400, 300, 300)
    w.setWindowTitle("CodersLegacy")
 
    combo = QtWidgets.QComboBox(w)
    combo.addItems(["Python", "Java", "C++"])
    combo.move(100, 100)
 
    button = QtWidgets.QPushButton(w)
    button.setText("Submit")
    button.clicked.connect(show)
    button.clicked.connect(show2)
    button.move(100, 200)

    grid = QtWidgets.QGridLayout(w)
    grid.addWidget(QPushButton("Button one"),0,0)
    grid.addWidget(QPushButton("Button two"),0,1)
    grid.addWidget(QPushButton("Button three"),0,2)

    line = QtWidgets.QLineEdit(w)
    line.setFixedWidth(140)
    line.move(100, 180)
    

    w.show()
    #w.show2()
    sys.exit(app.exec_())
"""

"""
def show_popup():
    msg = QMessageBox(win)
    msg.setWindowTitle("Message Box")
    msg.setText("Information about how to enter data in the simulation.")
    msg.setIcon(QMessageBox.Question)
     
    msg.setStandardButtons(QMessageBox.Cancel|QMessageBox.Ok)
    msg.setDefaultButton(QMessageBox.Ok)
 
    msg.setDetailedText("Information to be taken into account:\n"
                "\n"
                "To exit the box help and be able to enter the data, you\n"
                " must the click the tab 'X'.\n"
                "\n"
                "Different steps:\n"
                "Step 1: Insert the value of the frequency. Use a dot ( . ),\n"
                "instead of a coma ( , ).\n" 
                "Step 2: Insert the value of the amplitude. Use a dot ( . ),\n"
                "instead of a coma ( , ).\n"
                "Step 3: Insert the type of the signal. The different types\n"
                "are:\n"
                "Sine, Square, Triangle, RampUp, RampDown and DCVoltage.\n"
                "\n"
                "To introduce values, we need to select the panel and erase\n"
                "the quote. After that, we can introduce the value of the\n"
                "frequency, amplitude and type of signal. Once we have\n"
                "entered the data, we press the button 'Results'\n"
                "and we can see the results obtained. \n"
                "\n"
                "If we want to change the value, we just need to modified\n"
                "the values and press again the buttom. If we change\n"
                "values, the results will be different. Finally, if we don't\n"
                "press the the button, the image will dissapear in the\n"
                "following 60 seconds.\n")
    msg.setInformativeText("This is some extra informative text about how to enter data in the interface.")
    x = msg.exec_()
 
app = QApplication(sys.argv)
win = QMainWindow()
win.setGeometry(400,400,300,300)
win.setWindowTitle("CodersLegacy")
 
button = QtWidgets.QPushButton(win)
button.setText("Information")
button.clicked.connect(show_popup)
button.move(100,100)
 
win.show()
sys.exit(app.exec_())
"""

"""
def show_popup():
    msg = QMessageBox(win)
    msg.setWindowTitle("Message Box")
    msg.setText("This is some random text")
    msg.setIcon(QMessageBox.Question)
    msg.setStandardButtons(QMessageBox.Cancel|QMessageBox.Ok
                          |QMessageBox.Retry)
    msg.setInformativeText("This is some extra informative text")
    x = msg.exec_()
"""

"""
class GroupBox(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.setWindowTitle("GroupBox")
        layout = QGridLayout()
        self.setLayout(layout)

        groupbox = QGroupBox("GroupBox Example")
        groupbox.setCheckable(True)
        layout.addWidget(groupbox)
        
        vbox = QVBoxLayout()
        groupbox.setLayout(vbox)

        radiobutton = QRadioButton("RadioButton 1")
        vbox.addWidget(radiobutton)
        
        radiobutton = QRadioButton("RadioButton 2")
        vbox.addWidget(radiobutton)

        radiobutton = QRadioButton("RadioButton 3")
        vbox.addWidget(radiobutton)

        radiobutton = QRadioButton("RadioButton 4")
        vbox.addWidget(radiobutton)

        groupChannelB = QGroupBox("ChannelB")
        groupChannelB.setCheckable(True)
        layout.addWidget(groupChannelB)

        groupChannelB.setLayout(vbox)

        frequency = QLineEdit("INSERT")
        vbox.addWidget(frequency)

        
        
app = QApplication(sys.argv)
screen = GroupBox()
screen.show()
sys.exit(app.exec_())

"""


from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

#from Distances import radial_distance
def radial_distance(body1, body2, utc, ref, abcorr, obs):
    x_1 = 1
    x_2 = 4
    y_1 = 5
    y_2 = 2
    z_1 = 7
    z_2 = 6

    d_rad = np.sqrt((x_2 - x_1)**2.0 + (y_2 - y_1)**2.0 + (z_2 - z_1)**2.0)

    return d_rad

class Ui_window1(object):
    def setupUi(self, window1):
        window1.setObjectName("window1")
        window1.resize(485, 530) # 820 530
        self.centralwidget = QtWidgets.QWidget(window1)
        self.centralwidget.setObjectName("centralwidget")
        window1.setCentralWidget(self.centralwidget)

        self.groupBox_2 = QtWidgets.QGroupBox("groupBox_2", self.centralwidget)

        self.output_rd = QtWidgets.QTextBrowser(self.groupBox_2)
        self.output_rd.setGeometry(QtCore.QRect(10, 90, 331, 111))
        self.output_rd.setObjectName("output_rd")



        self.retranslateUi(window1)

        QtCore.QMetaObject.connectSlotsByName(window1)        

    def retranslateUi(self, window1):
            _translate = QtCore.QCoreApplication.translate
            window1.setWindowTitle(_translate("window1", "GUI"))


    def rad_distance(self):
        time_rd = np.asarray([1, 2])         # ? (self.get_time_rd())

        body1, body2 = ['EARTH', 'SUN']

        rad_dis = radial_distance(body1, body2, time_rd, 'HCI', 'NONE', 'SUN')

        #self.output_rd.setText(rad_dis)
        self.output_rd.append(str(rad_dis))                                     


if __name__ == "__main__":
    
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window1 = QtWidgets.QMainWindow()
    ui = Ui_window1()
    ui.setupUi(window1)
    
    #while True:
        #ui.rad_distance()
        #window1.show()
    
    ui.rad_distance()
    window1.show()
    sys.exit(app.exec_())


"""
lista = [406, 1018, 1626, 2239, 2847, 3460]
new_list = []
print(lista)
for i in range(len(lista)):
    new_list.append(lista[i] + 5)
print(new_list)
"""

"""
points = []
x = 0
for i in range(2000):
    x += 1
    if i > 9:
        del points[0]
        points.append(i* x * (1-x))
        print(points)
    else:
        points.append(i * x * (1-x))
        print(points)
"""

"""
cont = 0
suma = []
valor = 0
try:
    while True:
        cont += 1
        valor += 1
        if cont > 9:
            del suma[0]
            suma.append(valor)
        else:
            suma.append(valor)
        print(suma)
except KeyboardInterrupt:
    pass
"""      
