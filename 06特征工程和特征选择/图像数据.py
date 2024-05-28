

def a():
    # 如果仅仅是对图像进行向量化(抽取特征)那么只需要移除最后一程，使用前一层的输出即可
    from keras.applications.resnet50 import ResNet50, preprocess_input
    from keras.preprocessing import image
    from scipy.misc import face
    
    import numpy as np
    # 加载keras中的预训练网络，且移除最后一层
    model = ResNet50(weights='imagenet', include_top=False)
    # 加载图像并进行预处理
    model.summary()
    img = image.array_to_img(face())
    
    from PIL import Image
    # 查看浣熊图片
    # Image.Image.show(img)
    
    img = img.resize((224,224))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    # 获取特征向量
    features = model.predict(x)
    print(features)
a()