"""
Графический интерфейс для "охранника"
В окне показываются все проходящие люди с уровнями доступа
"""
import sys
import os
import psutil

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog, QMessageBox
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect
import cv2
import numpy as np
from multiprocessing import Process, Queue
from datetime import datetime

from gui_on_python import Ui_MainWindow
from select_cam_gui_on_python import Ui_Dialog
from utils import get_dict_of_valid_cams_id
from settings import CAMS_OF_DOORS as CAMS, FRAMES_COUNT
from pacs import RecognitionProcess


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.entry_videoThread = None
        self.exit_videoThread = None
        self.recognized_personsThread = None

        self.entry_input_queue = Queue(maxsize=5)
        self.exit_input_queue = Queue(maxsize=5)

        self.recognized_queue = Queue(maxsize=15)

        self.entry_recognitionProcess = None
        self.exit_recognitionProcess = None

        self.display_width = 556  # self.ui.entry_door_image_label.width()
        self.display_height = 529  # self.ui.exit_door_image_label.height()

        self.entry_frames_count = 0
        self.exit_frames_count = 0

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
        print(valid_cams)
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

        self.entry_recognitionProcess = RecognitionProcess(
            name='entry',
            image_queue=self.entry_input_queue,
            output_queue=self.recognized_queue,
        )
        self.recognized_personsThread = RecognisedThread(self.recognized_queue)
        self.recognized_personsThread.change_logged_signal.connect(
            self.update_last_logged
        )

        if CAMS['entry'] != CAMS['exit']:
            self.exit_videoThread = VideoThread(cam_id=CAMS['exit'])
            self.exit_videoThread.change_pixmap_signal.connect(
                self.update_image_exit)
            self.exit_videoThread.error_read_cam_signal.connect(
                self.show_error)
            self.exit_recognitionProcess = RecognitionProcess(
                name='exit',
                image_queue=self.exit_input_queue,
                output_queue=self.recognized_queue
            )

        self.entry_videoThread.start()
        self.entry_recognitionProcess.start()
        self.recognized_personsThread.start()
        if self.exit_videoThread:
            self.exit_videoThread.start()
            self.exit_recognitionProcess.start()

    def put_entry_frame(self, frame: np.ndarray) -> None:
        if self.entry_input_queue.full():
            self.entry_input_queue.get_nowait()
        self.entry_input_queue.put(frame)

    def put_exit_frame(self, frame: np.ndarray) -> None:
        if self.exit_input_queue.full():
            self.exit_input_queue.get_nowait()
        self.exit_input_queue.put(frame)

    @pyqtSlot(np.ndarray)
    def update_image_entry(self, cv_img: np.ndarray) -> None:
        """Updates the image_label with a new opencv image"""
        if self.entry_frames_count == FRAMES_COUNT:
            self.put_entry_frame(cv_img)
            self.entry_frames_count = 0

        self.entry_frames_count += 1

        qt_img = self.convert_cv_qt(cv_img)
        self.ui.entry_door_image_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_image_exit(self, cv_img: np.ndarray) -> None:
        """Updates the image_label with a new opencv image"""
        if self.exit_frames_count == FRAMES_COUNT:
            self.put_exit_frame(cv_img)
            self.exit_frames_count = 0

        self.exit_frames_count += 1

        qt_img = self.convert_cv_qt(cv_img)
        self.ui.exit_door_image_label.setPixmap(qt_img)

    def update_last_logged(self, door: str, name: str, image_path: str,
                           access: bool) -> None:
        """Отображение распознанных лиц на главном окне."""
        qt_img = self.convert_cv_qt(cv2.imread(image_path))
        time = datetime.now().strftime("%H:%M:%S")
        if door == 'entry':
            self.ui.logged_in_image_label.setPixmap(qt_img)
            self.ui.logged_in_name_label.setText(name)
            self.ui.logged_in_time_label.setText(time)
            self.ui.logged_in_access_label.setText(str(access))
        if door == 'exit':
            self.ui.logged_out_image_label.setPixmap(qt_img)
            self.ui.logged_out_name_label.setText(name)
            self.ui.logged_out_time_label.setText(time)

    def convert_cv_qt(self, cv_img: np.ndarray) -> QPixmap:
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

    def show_error(self, text: str) -> None:
        """Вывод ошибки при подключении к камере."""
        QMessageBox.critical(self, "Ошибка ", text, QMessageBox.Ok)

    def close_threads(self) -> None:
        """Завершение выполнения потоков воспроизведения видео и процессов."""
        if self.entry_videoThread:
            self.entry_videoThread.close_thread()
        if self.exit_videoThread:
            self.exit_videoThread.close_thread()
        if self.entry_recognitionProcess:
            self.entry_recognitionProcess.stop_process()
        if self.exit_recognitionProcess:
            self.exit_recognitionProcess.stop_process()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super(QMainWindow, self).closeEvent(a0)

        # Принудительное завершение всех дочерних процессов
        me = psutil.Process(os.getpid())
        for child in me.children():
            child.kill()
        self.close_threads()


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


class RecognisedThread(QThread):
    """
    Поток для отслеживания результатов распознования.
    Как только в выходной очереди процессов распознования появляется результат,
    подается сигнал на отображение родительскому потоку.
    """

    def __init__(self, logged_in_queue):
        super(RecognisedThread, self).__init__()

        self.logged_in_queue = logged_in_queue
        self.is_work = True

    change_logged_signal = pyqtSignal(str, str, str, bool)

    def run(self):
        while self.is_work:
            if not self.logged_in_queue.empty():
                logged_event = self.logged_in_queue.get()
                door = logged_event['door']
                person = logged_event['person']
                self.change_logged_signal.emit(
                    door,
                    person.name,
                    person.image_path,
                    person.access
                )

        self.close_thread()

    def close_thread(self):
        """Завершение потока."""
        self.is_work = False
        self.quit()


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

    mainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    application()
