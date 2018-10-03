import pickle
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv1D,MaxPooling1D,AveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras import optimizers
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize,TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer
import pymystem3
import re
from keras import regularizers
import keras
from IPython.display import clear_output
import json

class SentimentAnalyzer():
    def __init__(self, config_path):
        config = json.load(open(config_path))
        self.reg_power = config['reg_power']
        self.max_length = config['max_length']
        self.n_features = config['n_features']
        self.weights_path = config['weights_path']
        self.__build_model()
        self.cv=pickle.load(open(config['cv_path'],'rb'))
        self.proba_threshold = config['proba_threshold']

        pass

    def __char_vectorize_sentence(self, sent, countvecotrizer, max_length=50):
        sent = str(sent)
        n_chars = len(sent)
        seq = np.vstack([countvecotrizer.transform([c]).toarray() for c in sent])
        return seq

    def __make_padded_sequences(self, docs, max_length, cv=None):
        if cv == None:
            cv = CountVectorizer(analyzer='char', ngram_range=(1, 1)).fit(docs)
        vecs = [self.__char_vectorize_sentence(sent, self.cv, max_length) for sent in docs]
        seqs = np.array([np.pad(np.vstack(v), mode='constant', pad_width=((0, max_length - len(v)), (0, 0))) if len(v) < max_length else np.vstack(v)[:max_length, :] for v in vecs])
        return seqs, cv


    def __clean_tweet(self, text):
        return re.sub(pattern='https.+|\n|\\\\n|\"|^\d.\/|@\w+|\W', repl=' ', string=text).strip()

    def __build_model(self):
        net = Sequential()
        net.add(Conv1D(filters=750, kernel_size=2, input_shape=(self.max_length, self.n_features),
                       kernel_regularizer=regularizers.l2(self.reg_power), activation='relu',
                       kernel_initializer='he_uniform', trainable=False))
        net.add(Dropout(0.5))
        net.add(MaxPooling1D(4))
        net.add(Conv1D(filters=512, kernel_size=2, activation='relu', kernel_regularizer=regularizers.l2(self.reg_power),
                       kernel_initializer='he_uniform', trainable=False))
        net.add(Dropout(0.5))
        net.add(MaxPooling1D(4))
        net.add(Conv1D(filters=256, kernel_size=2, activation='relu', kernel_regularizer=regularizers.l2(self.reg_power),
                       kernel_initializer='he_uniform', trainable=False))
        net.add(Dropout(0.5))
        net.add(MaxPooling1D(4))
        net.add(Flatten())
        net.add(Dense(units=150, activation='relu', kernel_regularizer=regularizers.l2(1e-2), trainable=True))
        net.add(Dense(units=50, activation='relu', kernel_regularizer=regularizers.l2(1e-2), trainable=True))
        net.add(Dense(units=1, activation='sigmoid'))
        self.model = net
        adam = optimizers.Adam(lr=1e-4)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
        self.model.load_weights(self.weights_path)

    def run(self,text):
        text = [self.__clean_tweet(str(t).lower()) for t in text]
        seqs, _ = self.__make_padded_sequences(text, self.max_length, self.cv)
        proba = self.model.predict(seqs)
        decisions = ['positive' if p>self.proba_threshold else 'negative' for p in proba]
        probas = [p[0] if p>self.proba_threshold else 1-p[0] for p in proba]
        return {
            'decision':decisions,
            'confidence':probas
        }



if __name__ == '__main__':
    sa = SentimentAnalyzer('config.json')
    test_mes = [
        'new definition of #hodl',
        'How can I trust you to keep my coins and not send them back. Scam!',
        'I just logged in through the app (couldn’t yesterday). All my stuff is fine. Obviously you don’t have to trust me but I’m not posting a screenshot from my phone right now. Hope it assuages some concern',
        'Was really hoping to buy #TRX at .03 last night hope that price holds or I\'m gng to be pretty pissed at u guys.'
    ]
    print(sa.run(test_mes))

