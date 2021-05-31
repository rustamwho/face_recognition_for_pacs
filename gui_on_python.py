# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'security_guard_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1720, 801)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.open_button = QtWidgets.QPushButton(self.centralwidget)
        self.open_button.setGeometry(QtCore.QRect(1450, 630, 241, 91))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.open_button.setFont(font)
        self.open_button.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.open_button.setObjectName("open_button")
        self.close_button = QtWidgets.QPushButton(self.centralwidget)
        self.close_button.setGeometry(QtCore.QRect(1150, 630, 241, 91))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.close_button.setFont(font)
        self.close_button.setStyleSheet("background-color: rgb(207, 0, 0);")
        self.close_button.setObjectName("close_button")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 40, 1111, 431))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.video_stream_horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.video_stream_horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.video_stream_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.video_stream_horizontalLayout.setSpacing(6)
        self.video_stream_horizontalLayout.setObjectName("video_stream_horizontalLayout")
        self.entry_door_image_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.entry_door_image_label.setFont(font)
        self.entry_door_image_label.setStyleSheet("background-color: rgb(199, 199, 199);")
        self.entry_door_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.entry_door_image_label.setObjectName("entry_door_image_label")
        self.video_stream_horizontalLayout.addWidget(self.entry_door_image_label)
        self.exit_door_image_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.exit_door_image_label.setEnabled(True)
        self.exit_door_image_label.setAutoFillBackground(False)
        self.exit_door_image_label.setStyleSheet("background-color: rgb(197, 197, 197);")
        self.exit_door_image_label.setTextFormat(QtCore.Qt.AutoText)
        self.exit_door_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.exit_door_image_label.setObjectName("exit_door_image_label")
        self.video_stream_horizontalLayout.addWidget(self.exit_door_image_label)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 490, 551, 251))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.logs_verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.logs_verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.logs_verticalLayout.setSpacing(2)
        self.logs_verticalLayout.setObjectName("logs_verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.logs_verticalLayout.addWidget(self.label)
        self.textBrowser = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.textBrowser.setObjectName("textBrowser")
        self.logs_verticalLayout.addWidget(self.textBrowser)
        self.show_logs_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.show_logs_button.setFont(font)
        self.show_logs_button.setObjectName("show_logs_button")
        self.logs_verticalLayout.addWidget(self.show_logs_button)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 551, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(570, 10, 551, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.logged_in_image_label = QtWidgets.QLabel(self.centralwidget)
        self.logged_in_image_label.setGeometry(QtCore.QRect(1160, 40, 200, 250))
        self.logged_in_image_label.setStyleSheet("background-color: rgb(197, 197, 197);")
        self.logged_in_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.logged_in_image_label.setObjectName("logged_in_image_label")
        self.logged_in_name_label = QtWidgets.QLabel(self.centralwidget)
        self.logged_in_name_label.setGeometry(QtCore.QRect(1240, 300, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.logged_in_name_label.setFont(font)
        self.logged_in_name_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.logged_in_name_label.setObjectName("logged_in_name_label")
        self.logged_in_time_label = QtWidgets.QLabel(self.centralwidget)
        self.logged_in_time_label.setGeometry(QtCore.QRect(1240, 350, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.logged_in_time_label.setFont(font)
        self.logged_in_time_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.logged_in_time_label.setObjectName("logged_in_time_label")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1160, 300, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1160, 350, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.logged_out_image_label = QtWidgets.QLabel(self.centralwidget)
        self.logged_out_image_label.setGeometry(QtCore.QRect(1460, 40, 200, 250))
        self.logged_out_image_label.setStyleSheet("background-color: rgb(197, 197, 197);")
        self.logged_out_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.logged_out_image_label.setObjectName("logged_out_image_label")
        self.logged_out_time_label = QtWidgets.QLabel(self.centralwidget)
        self.logged_out_time_label.setGeometry(QtCore.QRect(1530, 350, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.logged_out_time_label.setFont(font)
        self.logged_out_time_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.logged_out_time_label.setObjectName("logged_out_time_label")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1460, 300, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(1460, 350, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.logged_out_name_label = QtWidgets.QLabel(self.centralwidget)
        self.logged_out_name_label.setGeometry(QtCore.QRect(1530, 300, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.logged_out_name_label.setFont(font)
        self.logged_out_name_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.logged_out_name_label.setObjectName("logged_out_name_label")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(600, 490, 201, 251))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.unknown_image_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.unknown_image_label.setStyleSheet("background-color: rgb(197, 197, 197);")
        self.unknown_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.unknown_image_label.setObjectName("unknown_image_label")
        self.verticalLayout.addWidget(self.unknown_image_label)
        self.save_unknown_button = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.save_unknown_button.setAutoFillBackground(False)
        self.save_unknown_button.setObjectName("save_unknown_button")
        self.verticalLayout.addWidget(self.save_unknown_button)
        self.delete_unknown_button = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.delete_unknown_button.setObjectName("delete_unknown_button")
        self.verticalLayout.addWidget(self.delete_unknown_button)
        self.count_logged_in_label = QtWidgets.QLabel(self.centralwidget)
        self.count_logged_in_label.setGeometry(QtCore.QRect(1010, 490, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.count_logged_in_label.setFont(font)
        self.count_logged_in_label.setObjectName("count_logged_in_label")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(870, 490, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(870, 520, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.count_logged_out_label = QtWidgets.QLabel(self.centralwidget)
        self.count_logged_out_label.setGeometry(QtCore.QRect(1010, 520, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.count_logged_out_label.setFont(font)
        self.count_logged_out_label.setObjectName("count_logged_out_label")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(1160, 390, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.logged_in_access_label = QtWidgets.QLabel(self.centralwidget)
        self.logged_in_access_label.setGeometry(QtCore.QRect(1240, 390, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.logged_in_access_label.setFont(font)
        self.logged_in_access_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.logged_in_access_label.setObjectName("logged_in_access_label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1720, 21))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menu_2)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.list_of_all_employees_action = QtWidgets.QAction(MainWindow)
        self.list_of_all_employees_action.setObjectName("list_of_all_employees_action")
        self.report_all_employees_by_day_action = QtWidgets.QAction(MainWindow)
        self.report_all_employees_by_day_action.setObjectName("report_all_employees_by_day_action")
        self.report_one_employee_by_day_action = QtWidgets.QAction(MainWindow)
        self.report_one_employee_by_day_action.setObjectName("report_one_employee_by_day_action")
        self.report_general_action = QtWidgets.QAction(MainWindow)
        self.report_general_action.setObjectName("report_general_action")
        self.add_employee_action = QtWidgets.QAction(MainWindow)
        self.add_employee_action.setObjectName("add_employee_action")
        self.delete_employee_action = QtWidgets.QAction(MainWindow)
        self.delete_employee_action.setObjectName("delete_employee_action")
        self.load_database_action = QtWidgets.QAction(MainWindow)
        self.load_database_action.setObjectName("load_database_action")
        self.clear_database_action = QtWidgets.QAction(MainWindow)
        self.clear_database_action.setObjectName("clear_database_action")
        self.set_cams_action = QtWidgets.QAction(MainWindow)
        self.set_cams_action.setObjectName("set_cams_action")
        self.exit_application_action = QtWidgets.QAction(MainWindow)
        self.exit_application_action.setObjectName("exit_application_action")
        self.select_recofnition_algorithm_action = QtWidgets.QAction(MainWindow)
        self.select_recofnition_algorithm_action.setObjectName("select_recofnition_algorithm_action")
        self.restart_recognition_action = QtWidgets.QAction(MainWindow)
        self.restart_recognition_action.setObjectName("restart_recognition_action")
        self.menu.addAction(self.list_of_all_employees_action)
        self.menu.addSeparator()
        self.menu.addAction(self.add_employee_action)
        self.menu.addAction(self.delete_employee_action)
        self.menu.addSeparator()
        self.menu.addAction(self.load_database_action)
        self.menu.addSeparator()
        self.menu.addAction(self.clear_database_action)
        self.menu_3.addAction(self.report_all_employees_by_day_action)
        self.menu_3.addAction(self.report_one_employee_by_day_action)
        self.menu_2.addAction(self.menu_3.menuAction())
        self.menu_2.addAction(self.report_general_action)
        self.menu_4.addAction(self.restart_recognition_action)
        self.menu_4.addAction(self.exit_application_action)
        self.menu_5.addAction(self.set_cams_action)
        self.menu_5.addAction(self.select_recofnition_algorithm_action)
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Автоматизированное рабочее место"))
        self.open_button.setText(_translate("MainWindow", "Открыть дверь"))
        self.close_button.setText(_translate("MainWindow", "Закрыть дверь"))
        self.entry_door_image_label.setText(_translate("MainWindow", "Входная дверь"))
        self.exit_door_image_label.setText(_translate("MainWindow", "Выходная дверь"))
        self.label.setText(_translate("MainWindow", "История событий"))
        self.show_logs_button.setText(_translate("MainWindow", "Посмотреть все логи"))
        self.label_2.setText(_translate("MainWindow", "Вход"))
        self.label_3.setText(_translate("MainWindow", "Выход"))
        self.logged_in_image_label.setText(_translate("MainWindow", "Последний вошедший"))
        self.logged_in_name_label.setText(_translate("MainWindow", "Имя"))
        self.logged_in_time_label.setText(_translate("MainWindow", "Время"))
        self.label_4.setText(_translate("MainWindow", "ФИО:"))
        self.label_5.setText(_translate("MainWindow", "Время:"))
        self.logged_out_image_label.setText(_translate("MainWindow", "Последний вышедший"))
        self.logged_out_time_label.setText(_translate("MainWindow", "Время"))
        self.label_6.setText(_translate("MainWindow", "ФИО:"))
        self.label_7.setText(_translate("MainWindow", "Время:"))
        self.logged_out_name_label.setText(_translate("MainWindow", "Имя"))
        self.unknown_image_label.setText(_translate("MainWindow", "Неизвестный посетитель"))
        self.save_unknown_button.setText(_translate("MainWindow", "Записать"))
        self.delete_unknown_button.setText(_translate("MainWindow", "Удалить"))
        self.count_logged_in_label.setText(_translate("MainWindow", "Количество"))
        self.label_8.setText(_translate("MainWindow", "В здании:"))
        self.label_9.setText(_translate("MainWindow", "Всего посещений:"))
        self.count_logged_out_label.setText(_translate("MainWindow", "Количество"))
        self.label_10.setText(_translate("MainWindow", "Доступ:"))
        self.logged_in_access_label.setText(_translate("MainWindow", "Доступ"))
        self.menu.setTitle(_translate("MainWindow", "База данных"))
        self.menu_2.setTitle(_translate("MainWindow", "Отчёт"))
        self.menu_3.setTitle(_translate("MainWindow", "По дням"))
        self.menu_4.setTitle(_translate("MainWindow", "Файл"))
        self.menu_5.setTitle(_translate("MainWindow", "Настройки"))
        self.list_of_all_employees_action.setText(_translate("MainWindow", "Список всех сотрудников"))
        self.report_all_employees_by_day_action.setText(_translate("MainWindow", "Все сотрудники"))
        self.report_one_employee_by_day_action.setText(_translate("MainWindow", "Отчет о сотруднике"))
        self.report_general_action.setText(_translate("MainWindow", "Общий"))
        self.add_employee_action.setText(_translate("MainWindow", "Добавить сотрудника"))
        self.delete_employee_action.setText(_translate("MainWindow", "Удалить сотрудника"))
        self.load_database_action.setText(_translate("MainWindow", "Загрузить базу данных"))
        self.clear_database_action.setText(_translate("MainWindow", "Очистить базу данных"))
        self.set_cams_action.setText(_translate("MainWindow", "Назначить камеры"))
        self.exit_application_action.setText(_translate("MainWindow", "Выход"))
        self.select_recofnition_algorithm_action.setText(_translate("MainWindow", "Выбрать алгоритм распознования"))
        self.restart_recognition_action.setText(_translate("MainWindow", "Запустить распознование"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
