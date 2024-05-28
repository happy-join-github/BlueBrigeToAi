import json
json_file = json.loads(open('ner.json','r',encoding='utf-8').read())
proess_data = []
for item in json_file:
    text = item['text']
    labels = ["O"] * len(text)
    
    if item['ann']:
        for annotation in item["ann"]:
            start = annotation["start"]
            end = annotation["end"]
            entity_label = annotation["label"]
            labels[start] = f"B-{entity_label}"
            for i in range(start + 1, end):
                labels[i] = f"I-{entity_label}"
            
    proess_data.append({
            "text": text,
            "text_id": item["text_id"],
            "label": labels
    })
    
with open('ner_processed.json', 'w', encoding='utf-8') as f:
    json.dump(proess_data,f,ensure_ascii=False)