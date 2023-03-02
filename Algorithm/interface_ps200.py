from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QGroupBox,
    QLineEdit, QPushButton, QSizePolicy, QWidget, QVBoxLayout, QGridLayout, QMainWindow)

import sys

class Form(QDialog):

    def __init__(self, parent = None):
        super(Form, self).__init__(parent)

        self.setWindowTitle("Interface")
        layout2 = QGridLayout()
        self.setLayout(layout2)
        self.resize(728,709)

        groupbox_cA = QGroupBox("ChannelA")
        groupbox_cA.setGeometry(QRect(80,130,151,80))
        layout2.addWidget(groupbox_cA)
        
        groupbox_cB = QGroupBox("ChannelB")
        layout2.addWidget(groupbox_cB)
        groupbox_cB.setGeometry(QRect(80, 250, 151, 80))

        groupbox_WP = QGroupBox("Wave properties")
        layout2.addWidget(groupbox_WP)
        groupbox_WP.setGeometry(QRect(10, 120, 68, 22))

        groupbox_DP = QGroupBox("Data to process")
        layout2.addWidget(groupbox_DP)
        groupbox_DP.setGeometry(QRect(80, 350, 571, 171))

        groupbox_FM = QGroupBox("Function mode")
        layout2.addWidget(groupbox_FM)
        groupbox_FM.setGeometry(QRect(80, 530, 561, 151))
    
        
        vbox = QVBoxLayout()
        groupbox_cA.setLayout(vbox)
        groupbox_cB.setLayout(vbox)
        groupbox_WP.setLayout(vbox)
        groupbox_DP.setLayout(vbox)
        groupbox_FM.setLayout(vbox)
        
        
        #Ejemplo.
        self.number_instants = QLineEdit("Insert")
        layout = QVBoxLayout()
        layout.addWidget(self.number_instants)
        self.setLayout(layout)
        

if __name__ == '__main__':

    app = QApplication(sys.argv)

    form = Form()
    form.show()
