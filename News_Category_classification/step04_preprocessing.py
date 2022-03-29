import pickle
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical (버젼이 다를 경우)

from konlpy.tag import Okt

# pip install konlpy 로 설치. https://webnautes.tistory.com/1394 설치방법 참조.
# 1번 링크에서 java8  win64 설치 고대로 next 설치 완료.
# 윈도우버튼 우클 시스템 > 고급 시스템 설정 > 고급 > 환경변수 설정 >
# Path 변수 > 편집 > 오라클/자바/자바패스 확인

# pip install JPype1-1.3.0-cp37-cp37m-win_amd64.whl
# pip install tweepy==3.10.0


pd.set_option("display.unicode.east_asian_width", True)
df = pd.read_csv("./crawling/special_crawling/total_naver_news.csv")
print(df.head())
print(df.info())

X = df["title"]
Y = df["category"]


### 라벨 엔코딩 & 원핫 엔코딩
encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)   # encoder : Y의 어떤 값을 어떤 숫자로 가지고 있는지 저장.
label = encoder.classes_
print(labeled_Y[0])
print(label)

with open("./models/special_encoder.pickle", "wb") as f:   # 피클 이용해서 encoder 저장.
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)


### 형태소 분리
okt = Okt()    # 오픈 코리안 텍스트
print(type(X))


for i in range(len(X)) :
    X[i] = okt.morphs(X[i], stem=True)   # 모든 X를 형태소로 나눠서 X에 넣는다.
print(X)


### 필요없는 단어들을 모은 stopwords.cs
stopwords = pd.read_csv("./crawling/stopwords.csv", index_col=0)

for j in range(len(X)) :
    words = []
    for i in range(len(X[j])) :
        if len(X[j][i]) > 1 :   # 길이가 1보다 큰 형태소만...
            if X[j][i] not in list(stopwords["stopword"]) :   # 불용어 제거.
                words.append(X[j][i])
    X[j] = " ".join(words)

print(X)


### 정수인코딩 : X에 있는 모든 단어를 숫자의 형태로 바꾸자.  with Tokenizer

token = Tokenizer()   # 단어 하나하나에 숫자를 부여한다. 정수 인코딩
token.fit_on_texts(X)   # token. 어떤 단어를 어떤 숫자로 부여했는지 정보를 가짐.
tokened_X = token.texts_to_sequences(X)  # token 딕셔너리를 이용해서 X를 토큰화 한다.

print(tokened_X[:5])

with open("./models/special_news_token.pickle", "wb") as f :
    pickle.dump(token, f)

wordsize = len(token.word_index) + 1     # 패딩을 위해 길이 1 추가.

print(wordsize)
print(token.index_word)


# 문장의 숫자를 맞춰야 하는데 빈 문장은 0으로 채워야함. 고로 문장의 길이가 제일 큰 놈 찾자.
max = 0
for i in range(len(tokened_X)) :
    if max < len(tokened_X[i]) :
        max = len(tokened_X[i])

print(max)

X_pad = pad_sequences(tokened_X, max)  # max 길이만큼 앞에 0을 붙여줌.
print(X_pad[:10])


X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save("./crawling/special_news_data_max_{}_wordsize_{}".format(max, wordsize), xy)

# 모델링을 colab으로 고고.  news_category_classification_03_learning.ipynb