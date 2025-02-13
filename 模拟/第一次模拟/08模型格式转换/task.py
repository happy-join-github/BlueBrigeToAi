# task-start
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512, num_classes=2):
        super(TextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, text):
        embedded = self.embedding(text)
        packed_output, (hidden, cell) = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


def convert():
    # TODO
    model = TextClassifier()
    model.load_state_dict(torch.load('model.pt'))
    x = torch.ones([256, 1], dtype=torch.long)
    # pt转onnx
    torch.onnx.export(
        model,
        x,
        'text_classifier.onnx',
        input_names=['input'],
        dynamic_axes={'input': {0: 'batch_size'}})


def inference(model_path, input):
    # TODO
    # 加载onnx模型文件
    session = ort.InferenceSession(model_path)
    # 处理输入的变量
    input_data = np.array(input, dtype=np.int64).reshape(-1, 1)
    # [101, 304, 993, 108, 102]
    # [[101],[304],[993],[108],[102]]
    result = session.run(None, {'input': input_data})[0]
    return result


def main():
    convert()
    result = inference('text_classifier.onnx', [101, 304, 993, 108, 102])
    print(result)


if __name__ == '__main__':
    main()
# task-end