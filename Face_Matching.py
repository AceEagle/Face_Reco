from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from numpy import asarray, ndarray
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image
from scipy.spatial.distance import cosine
import cv2


class FaceMatcher:
    def __init__(self):
        self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    def extract_face(self, filename, required_size=(224, 224)):
        if isinstance(filename, ndarray):
            pixel = filename
            print(pixel)
        else:
            pixel = plt.imread(filename)
            print(pixel)
        detector = MTCNN()
        result = detector.detect_faces(pixel)

        x1, y1, width, height = result[0]['box']
        x2, y2 = x1 + width, y1 + height

        face = pixel[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array


    def get_embeddings(self, filename):
        faces = [self.extract_face(f) for f in filename]
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = self.model.predict(samples)
        return yhat


    def is_match(self, known_embedding, candidate_embedding, tresh=0.3):
        score = cosine(known_embedding, candidate_embedding)
        if score <= tresh:
            print('>face is a Match (%.3f <= %.3f' % (score, tresh))
        else:
            print('>face is NOT a Match (%.3f > %.3f' % (score, tresh))

    def camera_match(self):
        filenames = ['Mat.jpg']
        embeddings = self.get_embeddings(filenames)
        Mat_id = embeddings[0]

        vc = cv2.VideoCapture(0)

        print('Positive Tests')
        for i in range(600):
            if vc.isOpened():
                rval, frame = vc.read()
                frameID = self.get_embeddings(asarray(frame, 'float32'))
                self.is_match(Mat_id, frameID)

a = FaceMatcher()

a.camera_match()