import os
import cv2
import numpy as np
import face_recognition

from utils import (save_data_as_pickle, load_data_from_pickle,
                   load_images_and_encoding)


def update_data(known_persons):
    save_data_as_pickle(known_persons)
    known_faces_encodings = []
    for person in known_persons:
        known_faces_encodings.append(person['face_encoding'])
    return known_faces_encodings


def main_loop(known_persons=[]):
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    known_faces_encodings = update_data(known_persons)
    while True:
        ret, frame = video_capture.read()

        # Quit when the input video file ends
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]

        face_location = face_recognition.face_locations(rgb_frame)
        cv2.imshow('Video', rgb_frame)
        if not face_location:
            print('None face')

            continue
        face_encoding = face_recognition.face_encodings(rgb_frame,
                                                        face_location)[0]

        face_distances = face_recognition.face_distance(
            known_faces_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.60:
            name_guest = known_persons[best_match_index]['name']
            access = known_persons[best_match_index]['access']
            image_path = known_persons[best_match_index]['face_path']
            print(f'Guest: {name_guest}\n'
                  f'Access: {access}\n'
                  f'Image path: {image_path}')
        else:
            print('Unknown guest!')

        # print(face_recognition.compare_faces(known_faces_encodings,face_encoding))
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    # Если база с лицами существует, загружем ее
    # Если баз нет, запускаем функцию распознавания по фоткам в директории
    if os.path.exists('known_peoples.pickle'):
        known_persons = load_data_from_pickle()
    else:
        known_persons = load_images_and_encoding()
    main_loop(known_persons)
