from datetime import datetime
from keras.utils import to_categorical
from konlpy.tag import Okt
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os, time, pickle
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def predict(filepath):
    if not filepath:
        print(f'path가 올바른 경로가 아니거나 입력하지 않았습니다...')
        exit()

    # 예측값을 알아보고 싶은 .csv 파일을 불러온다
    df = pd.read_csv(filepath)
    # 전처리 과정에서 저장했었던 encoder 파일을 불러온다
    with open('./pickled_ones/encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)
    # 불용 문자 파일(stopwords.csv)을 불러온다
    stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)
    # 토큰 파일을 불러온다
    with open('./pickled_ones/lyrics_token.pickle', 'rb') as f:
        token = pickle.load(f)
    # 학습된 모델 파일(.h5)을 불러온다
    model = load_model('./models/lyrics_category_model_0.5956.h5')


    X, Y = df['lyric'], df['category']
    labeled_Y = encoder.transform(Y)
    label = encoder.classes_
    onehot_Y = to_categorical(labeled_Y)


    ###  단어 형태소 분리
    okt = Okt()
    for i in range(len(X)):
        X[i] = okt.morphs(X[i], stem=True)

    ###  불용 문자 제거
    for i in range(len(X)):
        words = []
        for j in range(len(X[i])):
            if len(X[i][j]) > 1:
                if X[i][j] not in list(stopwords['stopword']):
                    words.append(X[i][j])
        X[i] = ' '.join(words)

    ###  단어 150개로 자르기
    words_list = []
    threshold = 150
    for i in range(len(X)):
        dummy = X[i].split()[:threshold]
        words = ''
        for j in range(len(dummy)):
            words = ' '.join([words, dummy[j]])

        words = words[1:]
        words_list.append(words)

    X = np.array(words_list)


    ###  토큰화 및 패딩작업
    tokened_X = token.texts_to_sequences(X)
    max_size = 0
    words_length_list = []
    for i in range(len(tokened_X)):
        words_length_list.append(len(tokened_X[i]))
        if len(tokened_X[i]) > max_size:
            max_size = len(tokened_X[i])

    X_pad = pad_sequences(tokened_X, max_size)

    ###  결과값 예측
    preds = model.predict(X_pad)
    predicts = []   # 모델이 예측한 레이블(최고확률)을 저장할 리스트
    pred_val = []   # 모델이 예측한 확률들을 저장할 리스트
    for pred in preds:
        predicts.append(label[np.argmax(pred)])
        pred_val.append(pred)

    # dataframe type is "pandas.DataFrame()"
    # predict_data type is "list"

    # 예측 데이터들(pred_val)을 3개까지 축소시켜 데이터프레임에 추가시키는 함수이다.
    df = reconstructPredictValue(dataframe=df, predict_data=pred_val, label=label)

    df["predict"] = predicts
    df["OX"] = 0

    # 예측을 잘못한 데이터만 따로 확인하기 위한 리스트 생성
    failed_list = []
    for i in range(len(df)):
        if df.loc[i, "category"] == df.loc[i, "predict"]:
            df.loc[i, "OX"] = "O"
        else:
            df.loc[i, "OX"] = "X"
            failed_list.append(df.loc[i, 'lyric'])

    debug_df = df[['category', 'predict', 'OX', 'predict_value']]
    print(debug_df)
    print('----------------------------------------')
    print(df["OX"].value_counts())
    print('----------------------------------------')

########################################################################
    ### OX_뒤의 숫자 (1,2,3)은  모델이 예측한 레이블이 상위 (1,2,3)에 포함되면 정답으로 카운팅한다.
    df["OX_1"] = 0
    df["OX_2"] = 0
    df["OX_3"] = 0
    for z in range(len(df["category"])) :
        # if (df["category"][z] == df['predict_value'][z][0]) or (df["category"][z] == df['predict_value'][z][2]) :
        #     df["OX_2"] = "O"
        # else : df["OX_2"] = "X"
        if df.loc[z, "category"] == df.loc[z, "predict_value"][0]:
            df.loc[z, "OX_1"] = "O"
        else:
            df.loc[z, "OX_1"] = "X"

        if df.loc[z, "category"] == df.loc[z, "predict_value"][0]:
            df.loc[z, "OX_2"] = "O"
        elif df.loc[z, "category"] == df.loc[z, "predict_value"][2]:
            df.loc[z, "OX_2"] = "O"
        else:
            df.loc[z, "OX_2"] = "X"

        if df.loc[z, "category"] == df.loc[z, "predict_value"][0]:
            df.loc[z, "OX_3"] = "O"
        elif df.loc[z, "category"] == df.loc[z, "predict_value"][2]:
            df.loc[z, "OX_3"] = "O"
        elif df.loc[z, "category"] == df.loc[z, "predict_value"][4]:
            df.loc[z, "OX_3"] = "O"
        else:
            df.loc[z, "OX_3"] = "X"

            # failed_list.append(df.loc[i, 'lyric'])

    print(df["OX_1"].value_counts("O"))
    print(df["OX_2"].value_counts("O"))
    print(df["OX_3"].value_counts("O"))



########################################################################

### 모델이 예측한 상위 (1,2,3)위의 레이블에   정답이 포함될 확률을 구하는 함수.

def reconstructPredictValue(dataframe, predict_data, label):

    copied_data = predict_data.copy()
    predict_result = []

    for i in range(len(copied_data)):
        contain_list = []
        # "copied_data[i]"을 백분율로 나타낸다.
        copied_data[i] = np.around(copied_data[i] * 100, 1)
        # "copied_data[i]"에서 총 3개의 값만 사용할 계획이므로 3번만 반복한다.
        # 예시. 리스트에서 높은 확률 순으로 레이블 추출.
        for j in range(3):
            # "copied_data[i]"에서 가장 큰 값의 인덱스 값을 "index"에 저장한다.
            index = np.argmax(copied_data[i])
            # "copied_data[i]"에서 가장 큰 값을 백분율로 표현하기 위해 '%'문자열과 합친다.
            percentage = ''.join([str(copied_data[i][index]), '%'])
            # "copied_data[i]"에서 가장 큰 값을 제거한다.
            copied_data[i] = np.delete(copied_data[i], index)
            # "copied_data[i]"에서 삭제된 데이터의 인덱스를 보존하기 위해 -0.1로 채워넣는다.
            copied_data[i] = np.insert(copied_data[i], index, -0.1)
            # "copied_data[i]"에서 가장 큰 값을 가진 인덱스(np.argmax())를 이용하여 카테고리 추가
            contain_list.append(label[index])
            # "copied_data[i]"에서 가장 큰 값을 "contain_list"에 추가
            contain_list.append(percentage)
        # 완성된 "contain_list"를 "predict_result"에 추가
        predict_result.append(contain_list)

    # 데이터프레임(df)에 "predict_value"라는 이름의 칼럼을 추가하며 칼럼의 값은 "predict_result"값으로 한다.
    dataframe['predict_value'] = predict_result
    return dataframe




if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    predict(filepath='data_for_predict/category_lyrics_total.csv')
