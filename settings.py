import os
from dotenv import load_dotenv

load_dotenv()

DIRECTORY_WITH_PHOTOS = 'C:/Dev/face_recognition_for_pacs/photos/'
DIRECTORY_FOR_FACES = 'C:/Dev/face_recognition_for_pacs/face_images/'

CAMS_OF_DOORS = {
    'entry': 0,
    'exit': 0,
}

FRAMES_COUNT = 50

DATABASES_SERVER = {
    'host': '127.0.0.1',
    'port': '5432',
    'user': os.getenv('POSTGRE_USERNAME'),
    'password': os.getenv('POSTGRE_PASS')
}
