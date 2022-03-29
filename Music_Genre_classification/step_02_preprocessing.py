from datetime import datetime
from keras.utils import to_categorical
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os, pickle, time
import pandas as pd


### 형태소 분리 (시간이 오래걸리므로 실행할 때 주의)
def splitMorphemes():
    # 실행 시간을 확인하기위해 선언하였다. (없어도 상관 없고, 지우려면 아래 runtime 부분도 지우면 된다.)
    start_time = time.time()
    # 모든 카테고리가 합쳐진 파일을 불러온다
    data = pd.read_csv('./crawling/lyrics_final.csv')  # type is 'DataFrame'

    # x에는 data의 'lyric'컬럼을 y에는 'category'컬럼을 할당한다.
    X, Y = data['lyric'], data['category']

    # 인코더는 카테고리(Ballad ~ Trot)를 정수로 변환해준다.
    # 발라드~트로트 총 7개의 카테고리가 존재하는데 이 데이터를 Ballad = 0, ..., Trot = 6 형태로 변환한다.
    encoder = LabelEncoder()
    labeled_Y = encoder.fit_transform(Y)
    # 사용하지 않고 있음. 지워도 무방할 것 같음.  --> 이 코드는 무엇을 의미하는지 확인해 볼 필요가 있음.
    label = encoder.classes_

    # encoder의 타입이 LabelEncoder() 타입이므로, 타입을 보존해야 하기 때문에 pickle을 이용하여 파일을 저장한다.
    with open('./pickled_ones/encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)

    # to_categorical(val)은 0~6 형태로 변환된 데이터를 one-hot 형태로 변환시킨다.
    # Ballad = 0 이므로 [1, 0, 0, 0, 0, 0, 0]
    # RnB_Soul = 3 이므로 [0, 0, 0, 1, 0, 0, 0]
    onehot_Y = to_categorical(labeled_Y)

    # one-hot 처리된 데이터는 numpy.ndarray 형태이므로 np.save()를 이용하여 해당 데이터를 저장한다. (궂이 pickle 모듈을 사용할 필요 없음)
    np.save('./saved_np/onehot_data.npy', onehot_Y)

    # Okt()란 Open Korean Text의 약자로, 한국어(자연어) 처리 모듈이다.
    okt = Okt()
    # lyric의 데이터가 담겨있는 x를 형태소 단위로 분리한다.
    # stem=True는 각 단어에서 어간을 추출하는 기능을 수행한다는 의미이다.
    # x의 각 원소(lyric)별로 형태소 단위로 분리하여 x의 데이터를 재구성한다.
    for i in range(len(X)):
        X[i] = okt.morphs(X[i], stem=True)

    # 형태소 단위로 분리된 x를 저장하기 위한 이름을 filename 에 할당된 문자열로 저장하려한다. (파일명을 적절히 수정하도록 한다.)
    filename = f'{datetime.now().strftime("%y%m%d_%H%M%S")}_okt_X.npy'
    # 형태소 단위로 분리된 x는 numpy.ndarray 형태이므로 np.save()를 이용하여 해당 데이터를 저장한다. (꼭 pickle 모듈을 사용할 필요 없음)
    np.save(f'./saved_np/{filename}', X)
    # 실행 시간을 확인하기위해 선언하였다. (없어도 상관 없고, 지우려면 위에 start_time 부분도 지우면 된다.)
    runtime = round(time.time() - start_time, 3)
    print(f'save to "{datetime.now().strftime("%y%m%d_%H%M%S")}_okt_X.npy"\nruntime is {runtime} seconds')
    # 40분 소요.

### 불용문자 제거
def removeUnusedWords():
    # 실행 시간을 확인하기위해 선언하였다. (없어도 상관 없고, 지우려면 아래 runtime 부분도 지우면 된다.)
    start_time = time.time()
    # 형태소 단위로 분리된 파일을 불러온다.
    # allow_pickle=True는 데이터의 타입을 보존하여 불러온다는 의미이다. (default 값은 True 이다.)
    okt_X = np.load('./saved_np/211120_151859_okt_X.npy', allow_pickle=True)    ####################
    # 불용문자(stopwords)가 포함되어있는 파일(.csv)를 불러온다.
    stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)

    # 형태소 단위로 분리된 모든 데이터(okt_x[0], okt_x[1], ...)에서 불용문자를 제거할 것이다.
    for i in range(len(okt_X)):
        words = []
        # okt_x의 i번째 인덱스에 들어있는 단어의 갯수만큼 반복한다. 반복하면 okt_x의 i번째 인덱스의 단어들을 1개씩 확인할 수 있다.
        for j in range(len(okt_X[i])):
            # okt_x의 i번째 인덱스의 j번째 단어의 길이가 1보다 길면 이하 실행. (단어가 있으면 실행되는 구조)
            if len(okt_X[i][j]) > 1:
                # okt_x의 i번째 인덱스의 j번째 단어가 stopwords에 들어있는 단어가 포함되어 있지 않으면 이하 실행
                # 즉, okt_x의 i번째 인덱스의 j번째 단어는 stopword가 없다는 것을 의미한다.
                if okt_X[i][j] not in list(stopwords['stopword']):
                    words.append(okt_X[i][j])
        # okt_x의 i번째 인덱스의 값은 불용문자를 제거한 나머지 단어들로만 구성하도록 한다.
        # 단어 자체가 아얘 없는 경우엔 공백으로 유지한다.
        okt_X[i] = ' '.join(words)

    # 불용문자가 제거된 okt_x는 numpy.ndarray 형태이므로 np.save()를 이용하여 해당 데이터를 저장한다. (궂이 pickle 모듈을 사용할 필요 없음)
    np.save('./saved_np/stopwords_okt_X.npy', okt_X)
    # 실행 시간을 확인하기위해 선언하였다. (없어도 상관 없고, 지우려면 위에 start_time 부분도 지우면 된다.)
    runtime = round(time.time() - start_time, 3)
    print(f'save to "stopwords_okt_X.npy"\nruntime is {runtime} seconds')     # 약 271초 소요


### stopwords_okt_X를 구성하는 원소들의 단어가 너무 많아서 슬라이싱 작업을 하기 위한 목적임
def sliceWords(**kwargs):
    # 단어 슬라이싱 된 결과를 확인하고 싶다면, debug를 True로 변경하면 된다.
    debug = True if 'debug' in kwargs else False
    # 실행 시간을 확인하기위해 선언하였다. (없어도 상관 없고, 지우려면 아래 runtime 부분도 지우면 된다.)
    start_time = time.time()
    # 불용 문자가 제거된 파일(stopwords_okt_X.npy)을 np.load()를 이용하여 불러온다.
    stopwords_okt_X = np.load('./saved_np/stopwords_okt_X.npy', allow_pickle=True)

    # 비어있는 리스트 생성
    words_list = []
    # lyric을 구성하고 있는 단어의 최대 갯수를 정의한다.
    threshold = 150
    # 불용 문자를 제거한 stopwords_okt_x를 구성하고 있는 단어의 길이가 너무 길어서 특정 부분까지만 수용하려고 한다.
    for i in range(len(stopwords_okt_X)):
        # threshold 개의 단어까지만 구성하고자 한다.
        dummy = stopwords_okt_X[i].split()[:threshold]
        # 비어있는 문자열 생성
        words = ''
        # 최대 150번까지 반복하며 dummy에서 단어 1개씩 words에 붙인다.
        for j in range(len(dummy)):
            # 붙일 때 공백문자(' ')를 포함하여 붙인다.
            words = ' '.join([words, dummy[j]])

        # 재구성 된 데이터들은 공통적으로 맨 앞에 공백(' ')이 포함되어 있다.
        # 공백은 0번 인덱스 이므로, 1번 인덱스부터 끝까지로 슬라이싱하여 재구성한다.
        # 즉, 맨 앞에 공백만 없애는 것이다.
        words = words[1:]
        # 맨 앞의 공백문자를 제거한 데이터를 words_list에 추가한다.
        words_list.append(words)

    # 완성된 words_list를 ndarray 형태로 변환한다. (원시데이터(stopwords_okt_x)의 타입이 numpy.ndarray 이기 때문이다.)
    result = np.array(words_list)
    # 각 자막(lyric)을 구성하고 있는 단어를 최대 150개까지로 재구성한 데이터(stopwords_okt_*_x.npy)를 저장한다.
    np.save(f'./saved_np/stopwords_okt_{threshold}_X.npy', result)

    # 실행 시간을 확인하기위해 선언하였다. (없어도 상관 없고, 지우려면 위에 start_time 부분도 지우면 된다.)
    runtime = round(time.time() - start_time, 3)
    print(f'save to "stopwords_okt_150_X.npy"\nruntime is {runtime} seconds')

    # debug module
    if debug:
        print('------------------------------------------------ DEBUG ------------------------------------------------')
        test1_len_list = []
        test2_len_list = []
        for i in range(len(stopwords_okt_X)):
            test1_len_list.append(len(list(map(str, stopwords_okt_X[i].split()))))
            test2_len_list.append(len(list(map(str, result[i].split()))))

        print(f'words length:\t{test1_len_list[19]}\nlyric words:\t{stopwords_okt_X[19]}')
        print(f'words length:\t{test2_len_list[19]}\nlyric words:\t{result[19]}')


# 단어 데이터를 토큰화하여 데이터셋을 구성하는 함수
def createToken():
    # 실행 시간을 확인하기위해 선언하였다. (없어도 상관 없고, 지우려면 아래 runtime 부분도 지우면 된다.)
    start_time = time.time()
    # 단어 길이에 제한을 두었던 파일(stopwords_okt_*_X.npy)을 np.load()를 이용하여 불러온다.
    stopwords_okt_X = np.load('./saved_np/stopwords_okt_150_X.npy', allow_pickle=True)
    # splitMorphemes() 함수에서 저장하였던 one-hot 데이터가 있는 파일을 np.load()를 이용하여 불러온다.
    onehot_Y = np.load('./saved_np/onehot_data.npy', allow_pickle=True)

    # 기존에 생성된 토큰 파일(lyrics_token.pickle)이 있으면 파일을 load한다.
    # 없는 경우에는 token을 생성하여 해당 token을 save한다.
    if 'lyrics_token.pickle' not in os.listdir('./pickled_ones'):
        token = Tokenizer()
        token.fit_on_texts(stopwords_okt_X)
        # 단어(word)를 Sequence 한다. 즉, 단어마다 번호를 지정한다.  --> 토큰화한다.
        tokened_X = token.texts_to_sequences(stopwords_okt_X)
        with open('./pickled_ones/lyrics_token.pickle', 'wb') as f:
            pickle.dump(token, f)
    else:
        with open('./pickled_ones/lyrics_token.pickle', 'rb') as f:
            token = pickle.load(f)
        # 단어(word)를 Sequence 한다. 즉, 단어마다 번호를 지정한다.  --> 토큰화한다.
        tokened_X = token.texts_to_sequences(stopwords_okt_X)

    # 토큰화 되어진 갯수를 파악한다.
    # 여러가지 단어가 있을 것이다. 각 단어들을 토큰화 시켰을 때 총 몇 개의 토큰정보가 있는지 확인하기 위함이다.
    # +1을 하는 이유는, 나중에 padding 작업을 통해 빈 공간은 '0'이라는 토큰을 사용하려고 +1을 하는 것이다.
    words_size = len(token.index_word) + 1
    print(words_size)
    # 확인 결과, 알파벳도 살아남은것을 확인하였음.
    # print(token.index_word)
    max_size = 0
    # print(token.index_word)
    print(len(tokened_X[0]), tokened_X[0])
    print(len(tokened_X[1]), tokened_X[1])
    print(len(tokened_X[19]), tokened_X[19])

    words_length_list = []
    # 토큰화(Tokenizing)된 데이터들을 확인한다.
    for i in range(len(tokened_X)):
        # 토큰화된 데이터의 길이를 words_length_list에 추가한다.
        words_length_list.append(len(tokened_X[i]))
        if len(tokened_X[i]) > max_size:
            max_size = len(tokened_X[i])

    print(f'max_size: {max_size}')

    # 토큰화 된 데이터(tokened_x)의 빈 공간을 0으로 채운다.
    # pad_sequences(토큰화된 데이터, max_size)는 가장 긴 토큰을 가지고 있는 데이터의 길이만큼 맞추어주고
    # 비어있는 공간을 0으로 채우는 작업을 수행한다.
    # 예를 들어, max_size=150이고, tokened_x[0]의 길이가 128이면, 150을 만들기 위해 22 만큼의 공간을 늘려주고
    # 늘린 공간을 0으로 채우는 작업을 수행한다. (단, 채우는 순서는 앞에서부터 채우게 된다.)
    x_pad = pad_sequences(tokened_X, max_size)

    # 토큰화된 데이터를 패딩 작업까지 마치면 학습 데이터셋(x_train, y_train)과 테스트 데이터셋(x_test, y_test)으로 분리한다.
    # x_pad가 입력이고, onehot_y가 정답이다. test_size=0.2는 전체 데이터(x_pad, onehot_y)의 20% 만큼 test-set 으로 사용하겠다는 의미.
    x_train, x_test, y_train, y_test = train_test_split(x_pad, onehot_Y, test_size=0.2)
    # xy 라는 이름을 가진 변수로 재구성
    xy = x_train, x_test, y_train, y_test

    # 완성된 데이터 셋(xy)을 저장한다.
    np.save(f'./saved_np/lyrics_dataset_{max_size}_{words_size}.npy', xy)

    # 실행 시간을 확인하기위해 선언하였다. (없어도 상관 없고, 지우려면 위에 start_time 부분도 지우면 된다.)
    runtime = round(time.time() - start_time, 3)
    print(f'save to "lyrics_dataset_{max_size}_{words_size}.npy"\nruntime is {runtime} seconds')


if __name__ == '__main__':      # 필요한 함수를 활성화하면 된다.
    pd.set_option("display.unicode.east_asian_width", True)
    # splitMorphemes()
    # removeUnusedWords()
    # sliceWords(debug=True)
    # createToken()
