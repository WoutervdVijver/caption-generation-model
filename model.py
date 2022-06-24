import numpy as np

from keras.utils import pad_sequences


class RnnModel:

    def __init__(self, model, vc, max_len):
        self.model = model
        self.vc = vc
        self.max_len = max_len

    def predict(self, image):
        desc = ['startseq']
        for i in range(36):   
            desc_tok = pad_sequences([self.vc(desc)], maxlen=self.max_len).reshape((1,36))
            pred = self.model.predict([image, desc_tok], verbose=0)
            pred_text = self.vc.get_vocabulary()[np.argmax(pred)]
            desc.append(pred_text)
        return desc