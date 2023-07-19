from keras.optimizers import RMSprop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.layers import Dense, Embedding, Flatten,MaxPooling1D,Conv1D, Dropout,GlobalMaxPooling1D, LSTM, Activation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
from tensorflow.keras.utils import to_categorical
from keras import Sequential
from keras.layers import LSTM, Embedding, Dropout, Conv1D, GlobalMaxPooling1D

okt = Okt()

#파일 경로 수정 
file_path = './mbti_result_final.csv'

#로드 및 파일 확인 
total_data  = pd.read_csv(file_path,encoding='cp949')
total_data = total_data[1:]
print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력
total_data['new_text'].nunique()
print('총 샘플의 수 :',len(total_data))

#결측값 확인 후 제거 
print(total_data.isnull().values.any())
total_data = total_data.dropna() # 결측값 제거aaaa
train_data = total_data
X_train = train_data['new_text'].values
y_train = train_data['labels'].values


# 사용하지 않음-----------------------------------------------------------------------------------------------------
# BoW / ※BoW, TF-IDF 중 하나 주석 처리 후 실행 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_train)
X = X.toarray()
X_pro_train = pad_sequences(X) # padding


# Tf-IDF / ※BoW, TF-IDF 중 하나 주석 처리 후 실행 
# vectorizer = TfidfVectorizer().fit(X_train)
# X = vectorizer.transform(X_train).toarray()
# X_pro_train = pad_sequences(X) # padding 
#-----------------------------------------------------------------------------------------------------



#train, test 분리 
X_train, X_test, y_train, y_test = train_test_split(X_pro_train, y_train, test_size = 0.2, random_state = 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#embedding_matrix = np.zeros((vocab_size,100))
embedding_dim = 100 # 임베딩 벡터의 차원
dropout_ratio = 0.3 # 드롭아웃 비율
num_filters = 256 # 커널의 수
kernel_size = 3 # 커널의 크기
hidden_units = 128 # 뉴런의 수
num_classes = 16


# 1. CNN-LSTM
model = Sequential()
model.add(Embedding(len(vectorizer.vocabulary_),100,input_length=X.shape[1]) )
model.add(MaxPooling1D())
model.add(Dropout(dropout_ratio))
model.add(Conv1D(128, 3, padding='same', strides=1))
model.add(Activation('relu'))
model.add(LSTM(64, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()



# 2. LSTM
# model = Sequential(name = 'LSTM')
# model.add(Embedding(len(vectorizer.vocabulary_), 100, input_length=X.shape[1]))
# model.add(Dropout(dropout_ratio))
# model.add(LSTM(units=hidden_units, recurrent_dropout=0.5, return_sequences = True))
# model.add(Dropout(dropout_ratio))
# model.add(Flatten())
# model.add(Dropout(dropout_ratio))
# model.add(Dense(num_classes, activation='softmax'))
# model.summary()




# 3. BiLSTM
# model = Sequential(name = 'BiLSTM')
# model.add(Embedding(len(vectorizer.vocabulary_), 100, input_length=X.shape[1]))
# model.add(Dropout(dropout_ratio))
# model.add(Bidirectional(LSTM(hidden_units)))
# model.add(Dropout(dropout_ratio))
# model.add(Dense(num_classes, activation='softmax'))
# print(model.summary())



# 4. CNN
# model = Sequential()
# model.add(Embedding(len(vectorizer.vocabulary_), 100, input_length=X.shape[1]))
# model.add(Dropout(dropout_ratio))
# model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(hidden_units, activation='relu'))
# model.add(Dropout(dropout_ratio))
# model.add(Dense(num_class, activation='softmax'))
# model.summary()


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=256, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))