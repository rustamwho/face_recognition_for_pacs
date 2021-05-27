"""
Графический интерфейс для "охранника"
В окне показываются все проходящие люди с уровнями доступа
"""
import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog, QMessageBox
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect
import cv2
import numpy as np

from gui_on_python import Ui_MainWindow
from select_cam_gui_on_python import Ui_Dialog
from utils import get_dict_of_valid_cams_id
from constants import CAMS_OF_DOORS as CAMS
from pacs import recognition_employees, load_data_for_recognition


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.entry_videoThread = None
        self.exit_videoThread = None

        self.entry_recognitionThread = None

        self.display_width = 556  # self.ui.entry_door_image_label.width()
        self.display_height = 529  # self.ui.exit_door_image_label.height()

        self.entry_frames_count = 0

        self.add_functions()

        self.start_recognition()

    def add_functions(self):
        """Назначение функций нажатию кнопок."""
        self.ui.restart_recognition_action.triggered.connect(
            self.start_recognition)
        self.ui.set_cams_action.triggered.connect(self.set_cams)

    def set_cams(self):
        """Выбор камер."""
        self.close_threads()

        # Получение списка доступных камер
        valid_cams = get_dict_of_valid_cams_id()
        if not valid_cams:
            self.show_error('Не найдено камер для подключения.')

        # Для каждой найденной камеры показать Диалоговое окно с выбором места
        for id_cam, cap in valid_cams.items():
            ret, cv_img = cap.read()
            if not ret:
                continue
            image = self.convert_cv_qt(cv_img)
            cap.release()
            dialog = DialogSetCam(image, id_cam)
            dialog.exec()

        self.start_recognition()

    def start_recognition(self):
        """Запуск потоков для отображения видеоряда."""
        # Завершение работающих потоков
        self.close_threads()

        self.entry_videoThread = VideoThread(cam_id=CAMS['entry'])
        self.entry_videoThread.change_pixmap_signal.connect(
            self.update_image_entry)
        self.entry_videoThread.error_read_cam_signal.connect(
            self.show_error)
        self.entry_recognitionThread = RecognitionThread(door='entry')

        if CAMS['entry'] != CAMS['exit']:
            self.exit_videoThread = VideoThread(cam_id=CAMS['exit'])
            self.exit_videoThread.change_pixmap_signal.connect(
                self.update_image_exit)
            self.exit_videoThread.error_read_cam_signal.connect(
                self.show_error)

        self.entry_videoThread.start()
        self.entry_recognitionThread.start()
        if self.exit_videoThread:
            self.exit_videoThread.start()

    @pyqtSlot(np.ndarray)
    def update_image_entry(self, cv_img):
        """Updates the image_label with a new opencv image"""
        if self.entry_frames_count == 25:
            self.entry_recognitionThread.detect_and_recognition_faces(cv_img)
            self.entry_frames_count = 0
        self.entry_frames_count += 1
        qt_img = self.convert_cv_qt(cv_img)
        self.ui.entry_door_image_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_image_exit(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.ui.exit_door_image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Конвертация OpenCV Image в QPixmap"""
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

    def show_error(self, text):
        """Вывод ошибки при подключении к камере."""
        QMessageBox.critical(self, "Ошибка ", text, QMessageBox.Ok)

    def close_threads(self):
        """Завершение выполнения потоков воспроизведения видео."""
        if self.entry_videoThread:
            self.entry_videoThread.close_thread()
        if self.exit_videoThread:
            self.exit_videoThread.close_thread()


class VideoThread(QThread):
    """Поток для отображения видеоряда."""

    def __init__(self, cam_id: int):
        super(VideoThread, self).__init__()
        self.cam_id = cam_id
        self.is_work = True

    error_read_cam_signal = pyqtSignal(str)
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(self.cam_id)
        while self.is_work:
            ret, cv_img = cap.read()
            if not ret:
                self.error_read_cam_signal.emit(
                    f'Невозможно подключиться к камере с id={self.cam_id}')
                cap.release()
                break

            self.change_pixmap_signal.emit(cv_img)

        self.close_thread()

    def close_thread(self):
        """Завершение потока."""
        self.is_work = False
        self.quit()


class RecognitionThread(QThread):
    def __init__(self, door: str = 'entry'):
        super(RecognitionThread, self).__init__()

        load_data_for_recognition()

    def detect_and_recognition_faces(self, image):
        recognition_employees(image)


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


if __name__ == '__main__':
    application()
