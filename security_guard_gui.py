"""
Графический интерфейс для "охранника"
В окне показываются все проходящие люди с уровнями доступа
"""
import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect
import cv2
import numpy as np

from gui_on_python import Ui_MainWindow
from select_cam_gui_on_python import Ui_Dialog
from utils import get_dict_of_valid_cams_id

CAMS = {
    'entry': 0,
    'exit': 0,
}


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.entry_videoThread = None
        self.exit_videoThread = None
        self.display_width = 556  # self.ui.entry_door_image_label.width()
        self.display_height = 529  # self.ui.exit_door_image_label.height()

        self.add_functions()

    def add_functions(self):
        """Назначение функций нажатию кнопок."""
        self.ui.restart_recognition_action.triggered.connect(
            self.start_recognition)
        self.ui.set_cams_action.triggered.connect(self.set_cams)

    def set_cams(self):
        """Выбор камер."""
        print('ras')
        valid_cams = get_dict_of_valid_cams_id()
        if not valid_cams:
            print("ERROR: Invalid cams")

        for id_cam, cap in valid_cams.items():
            ret, cv_img = cap.read()
            if not ret:
                continue
            image = convert_cv_qt(cv_img, 500, 500)

            dialog = DialogSetCam(image, id_cam)
            dialog.exec()

        print(CAMS)

    def start_recognition(self):
        """if CAMS['entry'] != CAMS['exit']:
            pass
        else:"""
        self.entry_videoThread = VideoThread()
        # connect its signal to the update_image slot
        self.entry_videoThread.change_pixmap_signal.connect(self.update_image)
        self.entry_videoThread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.ui.entry_door_image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h,
                                            bytes_per_line,
                                            QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width,
                                        self.display_height,
                                        Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(1)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)


class DialogSetCam(QtWidgets.QDialog):
    """Диалоговое окно для выбора камеры на вход и выход."""

    def __init__(self, image, id_cam):
        super(DialogSetCam, self).__init__()
        self.id_cam = id_cam

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.image_from_cam_label.setPixmap(image)
        self.ui.pushButton_ok.clicked.connect(self.save)

    def save(self):
        if self.ui.entry_door_radioButton.isChecked():
            CAMS['entry'] = self.id_cam
        if self.ui.exit_door_radioButton.isChecked():
            CAMS['exit'] = self.id_cam
        self.close()


def application():
    """Отрисовка окна"""
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    """MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    add_functions(ui)"""

    mainWindow.show()
    sys.exit(app.exec_())


def convert_cv_qt(cv_img, display_width, display_height):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h,
                                        bytes_per_line,
                                        QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(display_width, display_height,
                                    Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)


if __name__ == '__main__':
    application()
