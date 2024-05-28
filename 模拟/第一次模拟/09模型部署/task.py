#task-start
from flask import Flask, jsonify, request
import torch

app = Flask(__name__)
model = torch.jit.load('ner.pt')
model.eval()
index2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}

def process(inputs):
    # TODO
    outputs=  model(torch.tensor(inputs)).detach().numpy()
    results = []
    for sentence in outputs:
        result = []
        i=0
        while i<len(sentence):
            if index2label[sentence[i]][0]=='B':
                label = index2label[sentence[i]][2:]
                start = i
                i+=1
                while i<len(sentence) and index2label[sentence[i]]==f"I-{label}":
                    i+=1
                if i-start>1:
                    result.append({'end':i-1,"start":start,'label':label})
            else:
                i+=1
        results.append(result)
    return results


@app.route('/ner', methods=['POST'])
def ner():
    data = request.get_json()
    inputs = data['inputs']
    outputs = process(inputs)
    return jsonify(outputs)


if __name__ == '__main__':
    app.run(debug=True)
#task-end