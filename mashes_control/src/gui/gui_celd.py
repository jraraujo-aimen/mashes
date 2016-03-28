#!/usr/bin/python
# -*- coding: utf-8 -*-

# GUI for the celd
# Autor: Noemi Otero Carbon

import sys
from PyQt4 import QtCore, QtGui, uic

# Cargar nuestro archivo .ui
form_class = uic.loadUiType("gui_celd.ui")[0]


class QControl(QtGui.QWidget, form_class):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.btn_Mode.clicked.connect(self.btn_Mode_clicked)
        self.btn_Power.clicked.connect(self.btn_Power_clicked)

 # Event from btn_Mode button
    def btn_Mode_clicked(self):
        power = float(self.edit_Power.text())
        print power

 # Event from btn_Mode button
    def btn_Power_clicked(self):
        self.label_Power_detected.setText(self.edit_Power.text())

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    MyWindow = QControl(None)
    MyWindow.show()
    app.exec_()
