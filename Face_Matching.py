from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions
from numpy import expand_dims, asarray
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image
from scipy.spatial.distance import cosine


def extract_face(filename, required_size=(224, 224)):
    pixel = plt.imread(filename)

    detector = MTCNN()
    result = detector.detect_faces(pixel)

    x1, y1, width, height = result[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = pixel[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def get_embeddings(filenames):
    faces = [extract_face(f) for f in filenames]
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    yhat = model.predict(samples)
    return yhat


def is_match(known_embedding, candidate_embedding, tresh=0.5):
    score = cosine(known_embedding, candidate_embedding)
    if score <= tresh:
        print('>face is a Match (%.3f <= %.3f' % (score, tresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f' % (score, tresh))


filenames = ['Mat.jpg', 'Mat2.jpg', 'Mat3.jpg', 'Mat4.jpg', 'Mat5.jpg', 'Nico.jpg']
embeddings = get_embeddings(filenames)
Mat_id = embeddings[0]
print('Positive Tests')
is_match(embeddings[0], embeddings[1])
is_match(embeddings[0], embeddings[2])
is_match(embeddings[0], embeddings[3])
is_match(embeddings[0], embeddings[4])
print('Negative Tests')
is_match(embeddings[0], embeddings[5])
