import pickle
import numpy
import cv2
import json
import os
import face_recognition
from numpy import ndarray
from typing import Tuple, Union

DIRECTORY_WITH_PHOTOS = 'C:/Dev/face_recognition_for_pacs/photos/'
DIRECTORY_FOR_FACES = 'C:/Dev/face_recognition_for_pacs/face_images/'


def save_data_as_pickle(data: dict):
    """Запись базы известных лиц на диск в формате pickle."""
    with open('known_peoples.pickle', 'wb') as peoples_db:
        pickle.dump(data, peoples_db, protocol=pickle.HIGHEST_PROTOCOL)
    print('Known faces backed up to disk.')


def load_data_from_pickle() -> list:
    """Чтение базы известных лиц из диска в формате pickle."""
    with open('known_peoples.pickle', 'rb') as peoples_db:
        data = pickle.load(peoples_db)
    print('Known faces was loaded successfully.')
    return data


def save_data_as_json(data):
    """Запись базы известных лиц на диск в формате json."""
    with open('known_persons.json', 'w', encoding='utf-8') as peoples_db:
        json.dump(data, peoples_db)
    print('Known faces backed up to disk.')


def load_data_from_json() -> Tuple[dict]:
    """Чтение базы известных лиц из диска в формате json."""
    with open('known_persons.json', 'r') as peoples_db:
        data = json.load(peoples_db, encoding='utf-8')
    return data


def save_face_image(name: str, image: numpy.ndarray):
    """Сохранение изображения лица человека на диск."""
    file_name = DIRECTORY_FOR_FACES + name + '.jpg'
    cv2.imwrite(file_name, image)
    print(f'New face image of {name} saved on base directory.')
    return file_name


def load_images_and_encoding() -> list[dict[str, Union[str, ndarray, bool]]]:
    """Распознование и сохранение лиц по фотографиям в директории."""
    known_persons = []
    for photo in os.listdir(DIRECTORY_WITH_PHOTOS):
        image_rgb = face_recognition.load_image_file(
            DIRECTORY_WITH_PHOTOS + photo)
        person_name = photo[:-4]

        # Grab the image of the the face
        face_location = face_recognition.face_locations(image_rgb)[0]
        top, right, bottom, left = face_location
        face_image = image_rgb[top - 100:bottom + 120, left - 70:right + 70]
        face_image = face_image[:, :, ::-1]

        face_path = save_face_image(person_name, face_image)

        face_encoding = face_recognition.face_encodings(image_rgb)[0]
        access = True
        new_person = {
            'name': person_name,
            'face_encoding': face_encoding,
            'access': access,
            'face_path': face_path
        }
        known_persons.append(new_person)
    return known_persons

def get_dict_of_valid_cams_id():
    """Возрващает список из id доступных видеокамер."""
    # detect all connected webcams
    valid_cams = {}
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print('Warning: unable to open video source: ', i)
        else:
            valid_cams[i] = cap

    return valid_cams