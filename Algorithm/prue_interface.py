# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interfacebolpWD.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################
import PyQt5.QtWidgets
import PyQt5.QtGui

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QGroupBox,
    QLineEdit, QPushButton, QSizePolicy, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(728, 709)
        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(80, 130, 151, 80))
        self.comboBox = QComboBox(self.groupBox)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(10, 30, 68, 22))
        self.groupBox_2 = QGroupBox(Dialog)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(80, 250, 151, 80))
        self.comboBox_2 = QComboBox(self.groupBox_2)
        self.comboBox_2.setObjectName(u"comboBox_2")
        self.comboBox_2.setGeometry(QRect(10, 30, 68, 22))
        self.groupBox_3 = QGroupBox(Dialog)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(250, 120, 401, 211))
        self.lineEdit = QLineEdit(self.groupBox_3)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(10, 40, 311, 21))
        self.lineEdit_2 = QLineEdit(self.groupBox_3)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        self.lineEdit_2.setGeometry(QRect(10, 80, 311, 21))
        self.comboBox_3 = QComboBox(self.groupBox_3)
        self.comboBox_3.setObjectName(u"comboBox_3")
        self.comboBox_3.setGeometry(QRect(10, 120, 68, 22))
        self.lineEdit_3 = QLineEdit(self.groupBox_3)
        self.lineEdit_3.setObjectName(u"lineEdit_3")
        self.lineEdit_3.setGeometry(QRect(10, 170, 311, 21))
        self.groupBox_4 = QGroupBox(Dialog)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(80, 350, 571, 171))
        self.lineEdit_4 = QLineEdit(self.groupBox_4)
        self.lineEdit_4.setObjectName(u"lineEdit_4")
        self.lineEdit_4.setGeometry(QRect(10, 30, 461, 21))
        self.lineEdit_5 = QLineEdit(self.groupBox_4)
        self.lineEdit_5.setObjectName(u"lineEdit_5")
        self.lineEdit_5.setGeometry(QRect(10, 60, 461, 21))
        self.lineEdit_6 = QLineEdit(self.groupBox_4)
        self.lineEdit_6.setObjectName(u"lineEdit_6")
        self.lineEdit_6.setGeometry(QRect(10, 90, 461, 21))
        self.lineEdit_7 = QLineEdit(self.groupBox_4)
        self.lineEdit_7.setObjectName(u"lineEdit_7")
        self.lineEdit_7.setGeometry(QRect(10, 130, 461, 21))
        self.groupBox_5 = QGroupBox(Dialog)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(80, 530, 561, 151))
        self.pushButton = QPushButton(self.groupBox_5)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(20, 30, 241, 51))
        self.pushButton_2 = QPushButton(self.groupBox_5)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(280, 30, 261, 51))
        self.pushButton_3 = QPushButton(self.groupBox_5)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(190, 100, 181, 41))
        self.PushButton = QPushButton(Dialog)
        self.PushButton.setObjectName(u"PushButton")
        self.PushButton.setGeometry(QRect(80, 40, 571, 71))

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)


    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"Channel A - Input", None))
        self.comboBox.setCurrentText("")
        self.groupBox_2.setTitle(QCoreApplication.translate("Dialog", u"Channel B - Output", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Dialog", u"Wave's properties", None))
        self.lineEdit.setInputMask("")
        self.lineEdit.setText(QCoreApplication.translate("Dialog", u"Insert amplitude", None))
        self.lineEdit_2.setText(QCoreApplication.translate("Dialog", u"Insert frequency", None))
        self.lineEdit_3.setText(QCoreApplication.translate("Dialog", u"Voltage offset", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("Dialog", u"Data to processing", None))
        self.lineEdit_4.setText(QCoreApplication.translate("Dialog", u"Insert number of instants to create the measurement:", None))
        self.lineEdit_5.setText(QCoreApplication.translate("Dialog", u"Insert number of instant to analyze:", None))
        self.lineEdit_6.setText(QCoreApplication.translate("Dialog", u"Analyze all together:", None))
        self.lineEdit_7.setText(QCoreApplication.translate("Dialog", u"Insert number of polynomial degree:", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("Dialog", u"Function's mode", None))
        self.pushButton.setText(QCoreApplication.translate("Dialog", u"First mode", None))
        self.pushButton_2.setText(QCoreApplication.translate("Dialog", u"Second mode", None))
        self.pushButton_3.setText(QCoreApplication.translate("Dialog", u"End simulation", None))
        self.PushButton.setText(QCoreApplication.translate("Dialog", u"Information", None))


#setupUi()

#retranslateUi()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Dialog
    ui.setUpUi(Dialog)
    Dialog.show()


