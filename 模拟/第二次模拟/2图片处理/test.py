#task-start
import numpy as np 
import pandas as pd 
import json
import cv2
import os

def img_processor(data_path, dst_size = (224,224)):
    Image_std = [0.229, 0.224, 0.225]
    Image_mean = [0.485, 0.456, 0.406]
    _std = np.array(Image_std).reshape((1,1,3))
    _mean = np.array(Image_mean).reshape((1,1,3))
    image_src = cv2.imread(data_path)
    #TODO
    # 图像缩放至 (256, 256) 大小
    image = cv2.resize(image_src, (256, 256))

    # 获取缩放后图像的高度和宽度
    height, width = image.shape[:2]

    # 计算中心裁剪的起始坐标
    startx, starty = width // 2 - dst_size[0] // 2, height // 2 - dst_size[1] // 2

    # 对缩放后的图像进行中心裁剪
    image = image[startx:startx + dst_size[0], starty:starty + dst_size[1]]

    # 对图像进行标准化处理
    image = (image - _mean) / _std

    return image_src, image, (startx,starty)


def simple_generator(data_list, json_file, dst_size = (224, 224)):
    with open(json_file, 'r') as f:
        data = json.load(f)
    folder_map = {v[0]: (int(k), v[1]) for k,v in data.items()}

    for img_path in data_list:
        image_src, image, (startx,starty) = img_processor(img_path, dst_size)
        label = folder_map[img_path.split('/')[-2]]
        yield image_src, image, label, (startx,starty)

def main():
    Image_path = 'Imagedata/images'
    Json_path = 'Imagedata/image_class_index.json'

    data_list = []
    for dirname, _, filenames in os.walk(Image_path):
        if os.path.basename(dirname).startswith('n'):
            for filename in filenames:
                data_list.append(os.path.join(dirname, filename))

    # 创建生成器
    generator = simple_generator(data_list, Json_path)

    # 查看示例，检查图像、标签等属性的正确性
    num_samples = 5
    for _ in range(num_samples):
        image_src, image, label, (startx,starty) = next(generator)
        print("SrcImage shape:", image_src.shape)
        print("Image shape:", image.shape)
        print("Label:", label)
        print("startx and starty:",(startx,starty))

if __name__ == '__main__':
    main()
#task-end