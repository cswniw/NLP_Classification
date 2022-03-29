import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap      #이미지 출력을 위해
from PIL import Image    #이미지를 다루기 위해
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
form_window = uic.loadUiType('./app_for_lyrics.ui')[0]


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = load_model("./models/lyrics_category_model_0.5956.h5")
        self.pushButton.clicked.connect(self.let_us_predict)


    def let_us_predict(self) :
        given_words = self.plainTextEdit.toPlainText()

        with open('./pickled_ones/encoder.pickle', 'rb') as f:
            encoder = pickle.load(f)
        stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)
        with open('./pickled_ones/lyrics_token.pickle', 'rb') as f:
            token = pickle.load(f)

        label = encoder.classes_

        okt = Okt()
        X = okt.morphs(given_words, stem=True)
        # X = okt.morphs(df["lyric"],stem=True)
        words = []
        for i in X:
            if len(i) > 1 and i != "\n\n":
                if i not in list(stopwords['stopword']):
                    words.append(i)

        X = ' '.join(words)

        words_list = []
        threshold = 150

        dummy = X.split()[:threshold]
        words = ''
        for j in range(len(dummy)):
            words = ' '.join([words, dummy[j]])

        words = words[1:]
        words_list.append(words)

        X = np.array(words_list)

        tokened_X = token.texts_to_sequences(X)
        max_size = 150
        X_pad = pad_sequences(tokened_X, max_size)

        model = load_model("./models/lyrics_category_model_0.5956.h5")
        preds = model.predict(X_pad)
        predicts = (label[np.argmax(preds)])

        predicts = []
        pred_val = []
        for pred in preds:
            predicts.append(label[np.argmax(pred)])
            pred_val.append(pred)
        copied_data = pred_val.copy()
        predict_result = []
        for i in range(len(copied_data)):
            contain_list = []
            copied_data[i] = np.around(copied_data[i] * 100, 1)
            for j in range(3):
                index = np.argmax(copied_data[i])
                percentage = "".join([str(copied_data[i][index]), "%"])
                copied_data[i] = np.delete(copied_data[i], index)
                copied_data[i] = np.insert(copied_data[i], index, -0.1)
                contain_list.append(label[index])
                contain_list.append(percentage)
            predict_result.append(contain_list)
        df = pd.DataFrame()
        df["predict_value"] = predict_result

        first_genre = (df['predict_value'][0][0], df['predict_value'][0][1])
        second_genre = (df['predict_value'][0][2], df['predict_value'][0][3])
        third_genre = (df['predict_value'][0][4], df['predict_value'][0][5])

        self.label_1.setText(f'{first_genre[0]} ({first_genre[1]})')
        self.label_2.setText(f'{second_genre[0]} ({second_genre[1]})')
        self.label_3.setText(f'{third_genre[0]} ({third_genre[1]})')



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()

    sys.exit(app.exec_())