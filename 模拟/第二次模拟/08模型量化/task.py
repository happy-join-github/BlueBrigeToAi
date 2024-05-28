# quantize-start
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


def quantize_model(model_path, quantized_model_path):
    # TODO
    # 加载模型h5
    model = tf.keras.models.load_model(model_path)
    # 转换
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # 模型配置要求
    # 指定了TensorFlow Lite模型所支持的数据类型
    converter.target_spec.supported_types = [tf.float16]
    # 指定了TensorFlow Lite模型所支持的运算集合
    # 内置的运算符，如果某些运算符在TensorFlow Lite中没有对应的运算符，将回退到使用TensorFlow运算符
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # 和onnx差不多需要保存一个初始数据集
    rep_dataset = []
    for i in range(2):
        input_data = np.random.randint(0, 4, (1, 100))
        rep_dataset.append(input_data)
    # 设置初始数据集
    converter.representative_dataset = rep_dataset
    # 转换
    quantized_tfile_model = converter.convert()
    # 保存
    open(quantized_model_path, 'wb').write(quantized_tfile_model)


def prediction_label(test_sentence, model_path):
    # TODO
    # 加载词向量
    word_index = json.load(open('word_index.json', 'r'))
    # 创建转换器
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    
    # 将句子转换为序列
    test_seg = tokenizer.texts_to_sequences([test_sentence])
    # 将序列填充到指定的最大长度
    test_seq = pad_sequences(test_seg, maxlen=100)
    
    # 加载TFLite模型
    model = tf.lite.Interpreter(model_path=model_path)
    
    model.allocate_tensors() # 遗忘
    # 获取输入和输出张量的详细信息
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    # 使用模型进行预测
    model.set_tensor(input_details[0]['index'], test_seq.astype(np.float32))
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    # 阈值化
    prediction_label = (prediction > 0.5).astype(np.int_)[0][0]
    
    return prediction_label


def main():
    # 量化模型
    quantize_model('/home/project/model.h5', '/home/project/quantized_model.tflite')
    # 测试示例
    test_sentence = "一个 公益广告 ： 爸爸 得 了 老年痴呆  儿子 带 他 去 吃饭  盘子 里面 剩下 两个 饺子  爸爸 直接 用手 抓起 饺子 放进 了 口袋  儿子 愣住 了  爸爸 说  我 儿子 最爱 吃 这个 了  最后 广告 出字 ： 他 忘记 了 一切  但 从未 忘记 爱 你    「 转 」"
    print(prediction_label(test_sentence, '/home/project/quantized_model.tflite'))


if __name__ == "__main__":
    main()
# quantize-end