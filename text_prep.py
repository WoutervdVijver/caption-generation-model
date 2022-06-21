import joblib 
import os
import spacy

import regex as re

def load_descriptions(path):
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

def clean_descriptions(caption_dict):
    nlp = spacy.load('en_core_web_sm')
    clean_captions = {}
    for key in caption_dict.keys():
        clean_captions[key] = []
        for description in caption_dict[key]:
            doc = nlp(description)
            lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
            clean_captions[key].append(lemmas)
    return clean_captions

def to_vocabulary(captions):
    vocabulary = set()
    for key in captions.keys():
        [vocabulary.update(word) for word in captions[key]]
    return vocabulary

def load_dataset(path):
    with open(path, 'r') as file:
        doc = file.read()
    return [txt.rsplit('.jpg')[0] for txt in doc.split('\n')]

