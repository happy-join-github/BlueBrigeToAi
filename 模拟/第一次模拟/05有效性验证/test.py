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
    
    #TODO
    # 添加卷积层和池化层
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    # 添加全连接层
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))  # Dropout层

    # 添加输出层
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, epochs=100, batch_size=32, verbose=1,
              validation_split=0.1, shuffle=True, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    ])

    # 评估模型
    loss, accuracy = model.evaluate(train_images, train_labels, verbose=0)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # 保存模型
    if loss<1e-3:
        model.save('image_classify.h5')
    else:
        print('未达目标')
    




build_model_and_train()
#task-end