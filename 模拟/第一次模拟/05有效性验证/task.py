
#task-start
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def load_data():

    with open("dataset", "rb") as f:
        data = pickle.load(f, encoding="bytes")

        train_data = np.asarray(data[b'data'][:10])
        train_labels = data[b'labels'][:10]

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = to_categorical(train_labels)
    return train_data, train_labels


def build_model_and_train():
    train_images, train_labels = load_data()
    model = Sequential()

    # TODO
    def teacher():
        from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
        from tensorflow.keras.optimizers import Adam
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=Adam(lr=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, batch_size=128, epochs=50, validation_data=(train_images, train_labels))
        test_loss, test_acc = model.evaluate(train_images, train_labels)
        print(test_loss, test_acc)
        model_path = 'image_classify.h5'  # 模型保存路径和文件名
        model.save(model_path)
    def guanfang():
        from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
        from keras.metrics import accuracy
        model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
        model.add(MaxPool2D((2,2)))
        model.add(Flatten())
        model.add(Dense(10,activation='softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.fit(train_images,train_labels,epochs=100)
        model.save('image_classify.h5')

build_model_and_train()
#task-end