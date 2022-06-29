import numpy as np
from keras.layers import TextVectorization
from keras.models import Model
from keras.utils import pad_sequences
from typing import List, Dict, Tuple

def sequencer(dataset: List[str], captions: Dict[str, List[List[str]]], features: Dict[str, List[float]], vc: TextVectorization, max_len: int) -> Tuple[np.array, np.array, np.array]:
    ''' Function that takes the captions and features from a dataset of image IDs and and returns prepared data for the model
    
    :param dataset: list of strings of image IDs
    :param captions: dictionary of captions of images
    :param features: dictinary of features of images
    :param vc: keras.layes.TextVectorization to vectorize the captions
    :param max_len: int representing the length to which the captions need to be padded'''
    X_1, X_2, y = [], [], []
    for key in dataset:
        for desc in captions[key]:
            seq = ['startseq'] + desc + ['endseq']
            seq = vc(seq)
            for i in range(1, len(seq)):
                X_1.append(features[key][0])
                X_2.append(seq[:i])
                y.append(seq[i])
    X_2 = pad_sequences(X_2, maxlen=max_len+2)
    return np.array(X_1), X_2.reshape((X_2.shape[0],X_2.shape[1])), np.array(y)


def generate_seq(dataset: List[str], captions: Dict[str, List[List[str]]], features: Dict[str, List[float]], vc: TextVectorization, max_len: int) -> Tuple[List[np.array], np.array]:
    '''Generator that yields preprocessed arrays for captions features and targets to be put into the model
    
    :param dataset: list of strings of image IDs
    :param captions: dictionary of captions of images
    :param features: dictinary of features of images
    :param vc: keras.layes.TextVectorization to vectorize the captions
    :param max_len: int representing the length to which the captions need to be padded
    '''
    while True:
        for key in dataset:
            X_1, X_2, y = sequencer([key], captions, features, vc, max_len)
            yield [X_1, X_2], y


class RnnModel:
    ''' Class to store a model and a text vectorizer
    
    Attributes
    ----------

    model: keras.models.Model that represets the model
    vc: keras.layers.TextVectorization that vectorizes texts
    max_len: int that represents the maximum number of words of all captions in the dataset
    '''

    def __init__(self, model: Model, vocab: List[str], max_len: int):
        self.model = model
        self.vc = TextVectorization(
            vocabulary = ['startseq'] + vocab + ['endseq']
        )
        self.max_len = max_len

    def predict(self, image: List[float]) -> List[str]:
        '''
        Function returns prediction as a list of strings from a list of features of an image

        :param image:  list of floats representing features of an image
        '''
        desc = ['startseq']
        for i in range(self.max_len+2):   
            desc_tok = pad_sequences([self.vc(desc)], maxlen=self.max_len+2).reshape((1,self.max_len+2))
            pred = self.model.predict([image, desc_tok], verbose=0)
            pred_text = self.vc.get_vocabulary()[np.argmax(pred)]
            if pred_text == 'endseq':
                break
            desc.append(pred_text)
        return desc[1:]