import pickle

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical (버젼이 다를 경우)
# from keras.utils import to_categorical (버젼이 다를 경우)

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.max_columns", 20)

### 검증용 데이터 로드
df = pd.read_csv("./crawling/special_naver_headline_news_20220112.csv", index_col=0)

X = df["title"]
Y = df["category"]

### 엔코더 불러오기 & 타겟 라벨링
with open("./models/special_encoder.pickle", "rb") as f :
    encoder = pickle.load(f)

labeled_Y = encoder.transform(Y)
label = encoder.classes_

onehot_Y = to_categorical(labeled_Y)


### 형태소 분리, 한글자/불용어 제거

from konlpy.tag import Okt
okt = Okt()

for i in range(len(X)) :
    X[i] = okt.morphs(X[i], stem=True)

stopwords = pd.read_csv("./crawling/stopwords.csv", index_col=0)
for j in range(len(X)) :
    words = []
    for i in range(len(X[j])) :
        if len(X[j][i]) > 1 :
            if X[j][i] not in list(stopwords["stopword"]) :
                words.append(X[j][i])
    X[j] = " ".join(words)


### 토크나이저 불러오기.

with open("./models/special_news_token.pickle", "rb") as f :
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)

for i in range(len(tokened_X)) :    # 길이를 맞춰줘야한다.
    if 21 < len(tokened_X[i]) :
        tokened_X[i] = tokened_X[i][:21]

### 패딩
X_pad = pad_sequences(tokened_X, 21)


### 모델 불러오기
model = load_model("./models/special_news_category_classification_model_0.7514662742614746.h5")
preds = model.predict(X_pad)
predicts = []

for pred in preds :
    predicts.append(label[np.argmax(pred)])

df["predict"] = predicts

df["OX"] = 0
for i in range(len(df)) :
    if df.loc[i, "category"] == df.loc[i, "predict"] :
        df.loc[i, "OX"] = "O"
    else :
        df.loc[i, "OX"] = "X"
        print(df.loc[i,['title','category',"predict"]])

# print(df.head(30))
# print(df["OX"].value_counts()/len(df))
