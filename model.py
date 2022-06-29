import numpy as np
from keras.layers import TextVectorization
from keras.utils import pad_sequences


class RnnModel:

    def __init__(self, model, vocab, max_len):
        self.model = model
        self.vc = TextVectorization(
            vocabulary = ['startseq'] + vocab + ['endseq']
        )
        self.max_len = max_len

    def predict(self, image):
        desc = ['startseq']
        for i in range(self.max_len+2):   
            desc_tok = pad_sequences([self.vc(desc)], maxlen=self.max_len+2).reshape((1,self.max_len+2))
            pred = self.model.predict([image, desc_tok], verbose=0)
            pred_text = self.vc.get_vocabulary()[np.argmax(pred)]
            if pred_text == 'endseq':
                break
            desc.append(pred_text)
        return desc[1:]