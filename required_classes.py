class Person():
    def __init__(self, name, image_path, id=0, access=True, face_encoding = None):
        self.id = id
        self.name = name
        self.face_encoding = face_encoding
        self.image_path = image_path
        self.access = access