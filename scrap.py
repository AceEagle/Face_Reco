from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input, decode_predictions
from numpy import expand_dims, asarray, cosine
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
yhat = model.predict(samples)


def is_match(known_embedding, candidate_embedding, tresh=0.5):
    score = cosine(known_embedding, candidate_embedding)
    if score <= tresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, tresh))
    else:
        print('<face is NOT a Match (%.3f > %.3f)' % (score, tresh))