import joblib
import os

from keras.utils import load_img, img_to_array
from typing import Dict

from keras.applications.resnet import ResNet50, preprocess_input
from keras.models import Model


class Extractor:

    def __init__(self):
        self.model = ResNet50(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    def extract_features(self, directory:str)-> Dict[str, float]:
        '''
        Function that extract the features form all images in a directory using the ResNet50 neural network and stores the features in a dictionary

        :param directory: str that represents the path to directory with images
        '''
        
        features = {}
        for name in os.listdir(directory):
            image = load_img(directory + '/' + name, target_size=(224,224))
            image = img_to_array(image)
            image = image.reshape(1, image.shape[0], image.shape[1] ,image.shape[2])
            image = preprocess_input(image)
            prediction = self.model.predict(image, verbose=0)
            image_id = name.split('.')[0]
            features[image_id] = prediction
        return features

class Extractor_alt:

    def __init__(self):
        mod = ResNet50(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
        self.model = Model(inputs=mod.inputs, outputs=mod.layers[-2].output)

    def extract_features(self, directory:str)-> Dict[str, float]:
        '''
        Function that extract the features form all images in a directory using the ResNet50 neural network and stores the features in a dictionary

        :param directory: str that represents the path to directory with images
        '''
        
        features = {}
        for name in os.listdir(directory):
            image = load_img(directory + '/' + name, target_size=(224,224))
            image = img_to_array(image)
            image = image.reshape(1, image.shape[0], image.shape[1] ,image.shape[2])
            image = preprocess_input(image)
            prediction = self.model.predict(image, verbose=0)
            image_id = name.split('.')[0]
            features[image_id] = prediction
        return features

# extract = Extractor()
# features = extract.extract_features('data/Flicker8k_Dataset')
# print('Extracted Features: %d' % len(features))
# joblib.dump(features, 'features.pkl')