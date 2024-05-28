# task-start
import random

import pandas as pd
from torch.utils.data import Dataset


class MakeDataset(Dataset):
    
    def __init__(self):
        self.data = pd.read_csv('data.csv')[['text_id', 'text']].values
        self.locs = open('loc.txt', 'r', encoding='utf-8').read().split('\n')
        self.pers = open('per.txt', 'r', encoding='utf-8').read().split('\n')
    
    def __getitem__(self, item):
        text_id, text = self.data[item]
        text, aug_info = self.augment(text)
        return text_id, text, aug_info
    
    def __len__(self):
        return len(self.data)
    
    def augment(self, text):
        aug_info = {'locs': [], 'pers': []}
        
        # TODO
        for index, item in enumerate(self.locs):
            if item in text:
                num = random.randint(0, len(self.locs) - 1)
                while index == num:
                    num = random.randint(0, len(self.locs) - 1)
                
                aug_info['locs'].append({'original': item, 'replacement': self.locs[num]})
        for index, item in enumerate(self.pers):
            if item in text:
                num = random.randint(0, len(self.pers) - 1)
                while index == num:
                    num = random.randint(0, len(self.pers) - 1)
                
                aug_info['pers'].append({'original': item, 'replacement': self.pers[num]})
        
        for i in aug_info['locs']:
            text = text.replace(i['original'], i['replacement'])
        
        for j in aug_info['pers']:
            text = text.replace(j['original'], j['replacement'])
        
        return text, aug_info


def main():
    dataset = MakeDataset()
    for data in dataset:
        print(data)


if __name__ == '__main__':
    main()
# task-end
