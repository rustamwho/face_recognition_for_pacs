import pickle
import numpy
import cv2
import json
import os
import face_recognition
import psycopg2
from psycopg2 import Error
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from typing import Tuple, Union

from required_classes import Person
from settings import (DIRECTORY_WITH_PHOTOS, DIRECTORY_FOR_FACES,
                      DATABASES_SERVER, DATABASE_NAME, THRESHOLD)


def save_data_as_pickle(data: list[Person]):
    """Запись базы известных лиц на диск в формате pickle."""
    with open('known_peoples.pickle', 'wb') as peoples_db:
        pickle.dump(data, peoples_db, protocol=pickle.HIGHEST_PROTOCOL)
    print('Known faces backed up to disk.')


def load_data_from_pickle() -> list:
    """Чтение базы известных лиц из диска в формате pickle."""
    with open('known_peoples.pickle', 'rb') as peoples_db:
        data = pickle.load(peoples_db)

    known_persons = [person for person in data]

    print('Known faces was loaded successfully.')
    return known_persons


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


def load_images_and_encoding() -> list[Person]:
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
        new_person = Person(name=person_name,
                            face_encoding=face_encoding,
                            image_path=face_path,
                            access=access)
        res = add_employee_to_db(new_person)

        print(f'Сотрудник {person_name} добавлен в базу.')

        known_persons.append(new_person)
    return known_persons


def get_dict_of_valid_cams_id():
    """Возрващает список из id доступных видеокамер."""
    valid_cams = {}
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print('Warning: unable to open video source: ', i)
        else:
            valid_cams[i] = cap

    return valid_cams


# --------------------------РАБОТА С БАЗОЙ ДАННЫХ------------------------------

db_connection_pool: SimpleConnectionPool


def prepare_the_db():
    if database_exists_or_create():
        create_connection_pool_to_db()
        tables_exists_or_create()


def database_exists_or_create() -> bool:
    """
    Создание базы данных, если ее еще нет на сервере.
    :return: True if database exists or created, False if have a problem.
    """
    try:
        connection = psycopg2.connect(
            host=DATABASES_SERVER['host'],
            port=DATABASES_SERVER['port'],
            user=DATABASES_SERVER['user'],
            password=DATABASES_SERVER['password']
        )
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        # Курсор для выполнения операций с базой данных
        cursor = connection.cursor()

        sql_create_database = 'create database ' + DATABASE_NAME
        cursor.execute(sql_create_database)
    except psycopg2.DatabaseError as error:
        if (DATABASE_NAME + ' уже существет') or (
                DATABASE_NAME + ' is already exists') in error:
            print('База данных существует, можно подключаться')
        else:
            print('Ошибка при работе с PostgreSQL', error)
            return False
    finally:
        if connection:
            cursor.close()
            connection.close()
            print('Соединение с PostgreSQL закрыто')
    return True


def tables_exists_or_create() -> bool:
    """
    Создание таблиц.
    :param connection: Объект подключения к базе данных
    :return: True if tables exists or created, False if have a problem
    """
    with get_connection_to_db() as connection:
        try:
            cursor = connection.cursor()

            # Включение расшсирения CUBE для базы данных, если его еще нет
            cursor.execute('create extension if not exists cube;')

            # Cube может хранить <100 dimensions, а в encodings - 128
            # Поэтому делится на 2 части по 64 (vec_low, vec_high)
            # Создание таблицы сотрудников
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS employees (
                    id              SERIAL  PRIMARY KEY,
                    first_name      VARCHAR             NOT NULL,
                    last_name       VARCHAR             NOT NULL,
                    image_path      VARCHAR UNIQUE      NOT NULL,
                    access          BOOLEAN             NOT NULL,
                    vec_low         CUBE                NOT NULL,
                    vec_high        CUBE                NOT NULL
                    );
                    '''
            )
            # Создание индекса для векторов
            cursor.execute(
                '''
                create index if not exists vectors_vec_idx 
                on employees (vec_low, vec_high);
                '''
            )
            print('Таблица сотрудников создана.')
            connection.commit()
        except (Exception, Error) as error:
            print('Ошибка при работе с PostgreSQL', error)
            return False


def create_connection_pool_to_db():
    """Создание пула соединений к базе данных"""
    global db_connection_pool
    db_connection_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=DATABASES_SERVER['host'],
        port=DATABASES_SERVER['port'],
        user=DATABASES_SERVER['user'],
        password=DATABASES_SERVER['password'],
        database=DATABASE_NAME
    )
    if db_connection_pool:
        print('Пул соединений к бд создан')
        return True
    return False


def get_connection():
    """Получение подключения к базе данных."""
    connection_to_db = psycopg2.connect(host=DATABASES_SERVER['host'],
                                        port=DATABASES_SERVER['port'],
                                        user=DATABASES_SERVER['user'],
                                        password=DATABASES_SERVER[
                                            'password'],
                                        database=DATABASE_NAME)
    return connection_to_db


@contextmanager
def get_connection_to_db() -> psycopg2.extensions.connection:
    """
    Получение подключения к базе данных из пула.
    :return: Connection object to the database.
    """

    connection = db_connection_pool.getconn()
    try:
        yield connection
    except (Exception, Error) as error:
        print('Ошибка при получении соединения с PostgreSQL из пула:', error)
    finally:
        db_connection_pool.putconn(connection)


def add_employee_to_db(employee: Person):
    with get_connection_to_db() as connection:
        try:
            cursor = connection.cursor()
            v_low = ','.join(str(s) for s in employee.face_encoding[0:64])
            v_high = ','.join(str(s) for s in employee.face_encoding[64:128])
            name, surname = employee.name.split()
            query = f'''
                INSERT INTO employees(
                    first_name, last_name, image_path, access, vec_low,vec_high
                    ) VALUES (
                    '{name}',
                    '{surname}',
                    '{employee.image_path}',
                    '{employee.access}',
                    CUBE(array[{v_low}]),
                    CUBE(array[{v_high}])
                    ) ON CONFLICT DO NOTHING;
            '''
            cursor.execute(query)
            cursor.close()
            connection.commit()
            return True
        except (Exception, Error) as error:
            print('Ошибка при добавлении пользователя в базу:', error)
            connection.rollback()
            return False


def load_data_from_db()->list[Person]:
    """Чтение базы данных сотрудников из PostgreSQL."""
    known_persons = []
    connection = get_connection()
    try:
        cursor = connection.cursor()
        query = 'select * from employees'
        cursor.execute(query)
        for employee in cursor.fetchall():
            id = employee[0]
            name = employee[1]+' '+employee[2]
            image_path = employee[3]
            access = employee[4]
            encoding_str = employee[5][1:-1]+', '+employee[6][1:-1]
            encodings = numpy.fromstring(encoding_str,dtype=numpy.float64,sep=',')
            new_person = Person(
                name=name,
                image_path=image_path,
                access=access,
                face_encoding=encodings
            )
            known_persons.append(new_person)

        print('Данные загружены из базы.')

        return known_persons

    except (Exception, Error) as error:
        print('Ошибка при чтении из базы:', error)
        connection.rollback()
        return False


def recognise_employee(encodings):
    with get_connection_to_db() as connection:
        cursor = connection.cursor()
        query = "SELECT first_name, last_name FROM employees WHERE sqrt(power(CUBE(array[{}]) <-> vec_low, 2) + power(CUBE(array[{}]) <-> vec_high, 2)) <= {} ".format(
            ','.join(str(s) for s in encodings[0:64]),
            ','.join(str(s) for s in encodings[0][64:128]),
            THRESHOLD,
        ) + \
                "ORDER BY sqrt(power(CUBE(array[{}]) <-> vec_low, 2) + power(CUBE(array[{}]) <-> vec_high, 2)) <-> vec_high) ASC LIMIT 1".format(
                    ','.join(str(s) for s in encodings[0:64]),
                    ','.join(str(s) for s in encodings[64:128]),
                )
        print(cursor.query(query))


if __name__ == '__main__':
    if database_exists_or_create():
        create_connection_pool_to_db()
        tables_exists_or_create()
        load_images_and_encoding()
