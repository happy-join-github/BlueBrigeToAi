import json
json_data = json.loads(open('ner.json', 'r', encoding='utf-8').read())

def process_json_data(data):
    processed_data = []
    for entry in data:
        labels = ["O"] * len(entry["text"])  # 初始化标签列表，全部设为O

        # 如果有标注，更新标签
        if "ann" in entry and entry["ann"]:
            for annotation in entry["ann"]:
                start = annotation["start"]
                end = annotation["end"]
                entity_label = annotation["label"]
                labels[start] = f"B-{entity_label}"
                for i in range(start+1, end):
                    labels[i] = f"I-{entity_label}"

        # 添加到处理后的数据列表中
        processed_data.append({
            "text": entry["text"],
            "text_id": entry["text_id"],
            "label": labels
        })

    return processed_data
lst = process_json_data(json_data)
with open('ner_processed.json', 'w', encoding='utf-8') as f:
    json.dump(lst,f,ensure_ascii=False)
    
