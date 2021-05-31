import os
import cv2
import numpy as np
import face_recognition
import dlib
from multiprocessing import Process, Queue

from utils import (save_data_as_pickle, load_data_from_pickle,
                   load_images_and_encoding)

dlib.DLIB_USE_CUDA = True


class Person():
    def __init__(self, name, image_path, access):
        self.name = name
        self.image_path = image_path
        self.access = access


class RecognitionProcess(Process):
    def __init__(self, name: str, image_queue: Queue, output_queue: Queue):
        Process.__init__(self)
        self.name = name
        self.image_queue = image_queue
        self.is_work = True
        self.known_faces_encodings = None
        self.known_persons = None
        self.output_queue = output_queue

        self.load_data_for_recognition()

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
            self.known_persons = load_data_from_pickle()
        else:
            self.known_persons = load_images_and_encoding()
        self.update_data()

    def update_data(self):
        save_data_as_pickle(self.known_persons)
        self.known_faces_encodings = []
        for person in self.known_persons:
            self.known_faces_encodings.append(person['face_encoding'])

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
            name_guest = self.known_persons[best_match_index]['name']
            access = self.known_persons[best_match_index]['access']
            image_path = self.known_persons[best_match_index]['face_path']
            print(f'Guest: {name_guest}\n'
                  f'Access: {access}\n'
                  f'Image path: {image_path}')
            self.output_queue.put({'door': self.name,
                                   'name': name_guest,
                                   'access': access,
                                   'image_path': image_path})
        else:
            print('Unknown guest!')


if __name__ == "__main__":
    # Если база с лицами существует, загружем ее
    # Если баз нет, запускаем функцию распознавания по фоткам в директории
    if os.path.exists('known_peoples.pickle'):
        known_persons = load_data_from_pickle()
    else:
        known_persons = load_images_and_encoding()
    main_loop(known_persons)
