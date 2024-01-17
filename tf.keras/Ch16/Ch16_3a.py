﻿import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM

seed = 10
np.random.seed(seed)  # 指定亂數種子
# 載入 IMDb 資料集
top_words = 1000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(
                                num_words=top_words)
# 資料預處理
max_words = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# 定義模型
imdb_input = Input(shape=(100,), dtype="int32",
                   name="imdb_input")
embed = Embedding(top_words, 32, input_length=max_words,
                    name="embed")(imdb_input)
drop1 = Dropout(0.25, name="drop1")(embed)
lstm = LSTM(32, name="lstm")(drop1)
drop2 = Dropout(0.25, name="drop2")(lstm)
output = Dense(1, activation="sigmoid", 
               name="output")(drop2)
model = Model(inputs=imdb_input, outputs=output)
model.summary()   # 顯示模型摘要資訊
from tensorflow.keras.utils import plot_model

plot_model(model, to_file="Ch16_3a.png", show_shapes=True)
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="rmsprop", 
              metrics=["accuracy"])
# 訓練模型
history = model.fit(X_train, Y_train, validation_split=0.2, 
          epochs=5, batch_size=128, verbose=2)
# 評估模型
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
# 顯示訓練和驗證損失圖表
import matplotlib.pyplot as plt

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 顯示訓練和驗證準確度
acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



