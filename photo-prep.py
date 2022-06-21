import joblib
import os

import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array

from keras.applications.resnet import ResNet50, preprocess_input

def extract_features(directory):
    model = ResNet50(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    features = {}
    for name in os.listdir(directory):
        image = load_img(directory + '/' + name, target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape(1, image.shape[0], image.shape[1] ,image.shape[2])
        image = preprocess_input(image)
        prediction = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = prediction
    return features

features = extract_features('data/Flicker8k_Dataset')
print('Extracted Features: %d' % len(features))
joblib.dump(features, 'features.pkl')