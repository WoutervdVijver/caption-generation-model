import joblib 
from typing import List
import os
import spacy

import regex as re

def load_descriptions(path:str) -> dict:
    '''
    Function that loads all the descriptions of all images from a txt file and return a dictionary where the keys are the image IDs and the items the descriptions

    :param path: str that represents the path to the text file of descriptions
    '''
    with open(path) as file:
        text = file.read()

    captions = re.findall(r'(\w*).jpg#([0-9])\s(...*\.?)', text)

    caption_dict = {}
    for caption in captions:
        if caption[0] not in caption_dict.keys():
            caption_dict[caption[0]] = [caption[2]]
        else:
            caption_dict[caption[0]].append(caption[2])

    return caption_dict

def clean_descriptions(caption_dict: dict) -> dict:
    '''
    Function that cleans all captions in a dictionary
    
    :param caption_dict: dictionary that needs to be cleaned
    '''
    nlp = spacy.load('en_core_web_sm')
    clean_captions = {}
    for key in caption_dict.keys():
        clean_captions[key] = []
        for description in caption_dict[key]:
            doc = nlp(description)
            lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
            clean_captions[key].append(lemmas)
    return clean_captions

def to_vocabulary(captions: dict) -> List[str]:
    '''
    Function that a list of all words appearing in the given dictionary of descriptions
    
    :param captions: dictionary of descriptions
    '''
    vocabulary = set()
    for key in captions.keys():
        [vocabulary.update(word) for word in captions[key]]
    return list(vocabulary)

def load_dataset(path: str) -> List[str]:
    '''
    Function that return a list of image IDs
    
    :param path: str that represents path to txt file with image IDs
    '''
    with open(path, 'r') as file:
        doc = file.read()
    return [txt.rsplit('.jpg')[0] for txt in doc.split('\n') if txt!='']

