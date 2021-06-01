import os
import cv2
import numpy as np
import face_recognition
import dlib
from multiprocessing import Process, Queue

from utils import (save_data_as_pickle, load_data_from_pickle,
                   load_images_and_encoding)
from required_classes import Person

dlib.DLIB_USE_CUDA = True


class RecognitionProcess(Process):
    def __init__(self, name: str, image_queue: Queue, output_queue: Queue):
        Process.__init__(self)
        self.name = name
        self.image_queue = image_queue
        self.output_queue = output_queue
        self.is_work = True

        self.known_persons = self.load_data_for_recognition()
        self.known_faces_encodings = self.update_data()

    def get_image(self):
        if not self.image_queue.empty():
            return True, self.image_queue.get()
        else:
            return False, None

    def stop_process(self):
        self.is_work = False

    def run(self):
        while self.is_work:
            ret, image = self.get_image()
            if ret:
                self.recognition_employees(image)

    def load_data_for_recognition(self):
        if os.path.exists('known_peoples.pickle'):
            return load_data_from_pickle()
        else:
            return load_images_and_encoding()

    def update_data(self):
        save_data_as_pickle(self.known_persons)

        return [person.face_encoding for person in self.known_persons]

    def recognition_employees(self, image):
        face_location = face_recognition.face_locations(image)

        if not face_location:
            print('None face')
            return None

        face_encoding = face_recognition.face_encodings(image,
                                                        face_location)[0]

        face_distances = face_recognition.face_distance(
            self.known_faces_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.60:
            recognized_person = self.known_persons[best_match_index]
            print(f'Guest: {recognized_person.name}\n'
                  f'Access: {recognized_person.access}\n'
                  f'Image path: {recognized_person.image_path}')
            self.output_queue.put({'door': self.name,
                                   'person': recognized_person})
        else:
            print('Unknown guest!')


if __name__ == "__main__":
    # Если база с лицами существует, загружем ее
    # Если баз нет, запускаем функцию распознавания по фоткам в директории
    if os.path.exists('known_peoples.pickle'):
        known_persons = load_data_from_pickle()
    else:
        known_persons = load_images_and_encoding()
    # main_loop(known_persons)
