import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

X_train,X_test,Y_train,Y_test = np.load("./crawling/special_news_data_max_21_wordsize_12454.npy",
                                        allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(12454, 300, input_length=21))  # 단어의 갯수 wordsize 12454개의 차원에서 학습.
# 차원이 너무 많아지면 1개의 데이타가 희소해진다. 너무 적어도 안된다.
# 12454개의 차원을 300개 차원 수준으로 줄이자.
# 좌표와 크기를 가진 것이 벡터.

model.add(Conv1D(32, kernel_size=5, padding="same", activation="relu"))
# 1차원 컴브레이어.
model.add(MaxPool1D(pool_size=1))
# pool_size=1은 아무것도 안 준거랑 마찬가지. 하는게 없는 의미. 일단 세트로 Conv1D와 묶어 줌.
# 커널의 0,1값은 처음은 랜덤.. 이후 학습을 하면서 수정됨.

model.add(LSTM(128, activation="tanh", return_sequences=True))
# LSTM은 tanh 준다.   순서를 확인하기위해 LSTM 모델 사용.
# return_sequences : 저장 여부. 앞단의 LSTM의 출력값을 저장해서 그 다음의 LSTM으로 준다.
model.add(Dropout(0.3))

model.add(LSTM(64, activation="tanh", return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(64, activation="tanh"))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation="relu"))

model.add(Dense(6, activation="softmax"))
#출력 카테고리 수 = 6
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"]) # 다중 분류기 시 categorical_crossentropy
fit_hist = model.fit(X_train, Y_train, batch_size=100, epochs=20, validation_data=(X_test,Y_test))

model.save("./models/special_news_category_classification_model_{}.h5".format(
    fit_hist.history["val_accuracy"][-1]))

plt.plot(fit_hist.history["accuracy"], label="accuracy")
plt.plot(fit_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()