# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'select_cam_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(612, 492)
        self.image_from_cam_label = QtWidgets.QLabel(Dialog)
        self.image_from_cam_label.setGeometry(QtCore.QRect(30, 20, 560, 370))
        self.image_from_cam_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_from_cam_label.setObjectName("image_from_cam_label")
        self.entry_door_radioButton = QtWidgets.QRadioButton(Dialog)
        self.entry_door_radioButton.setGeometry(QtCore.QRect(230, 420, 71, 17))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.entry_door_radioButton.setFont(font)
        self.entry_door_radioButton.setObjectName("entry_door_radioButton")
        self.exit_door_radioButton = QtWidgets.QRadioButton(Dialog)
        self.exit_door_radioButton.setGeometry(QtCore.QRect(310, 420, 91, 17))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.exit_door_radioButton.setFont(font)
        self.exit_door_radioButton.setObjectName("exit_door_radioButton")
        self.pushButton_ok = QtWidgets.QPushButton(Dialog)
        self.pushButton_ok.setGeometry(QtCore.QRect(220, 450, 75, 23))
        self.pushButton_ok.setObjectName("pushButton_ok")
        self.pushButton_cancel = QtWidgets.QPushButton(Dialog)
        self.pushButton_cancel.setGeometry(QtCore.QRect(310, 450, 75, 23))
        self.pushButton_cancel.setObjectName("pushButton_cancel")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Выбор камеры"))
        self.image_from_cam_label.setText(_translate("Dialog", "Изображение"))
        self.entry_door_radioButton.setText(_translate("Dialog", "Вход"))
        self.exit_door_radioButton.setText(_translate("Dialog", "Выход"))
        self.pushButton_ok.setText(_translate("Dialog", "OK"))
        self.pushButton_cancel.setText(_translate("Dialog", "Отмена"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
