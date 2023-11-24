from urllib.request import urlopen
from PIL import Image
import timm
import torch
from tqdm import tqdm
import numpy as np
import os
from labels import label_get
from torchvision import models
import requests
from io import BytesIO
from img import image_process
import sklearn.cluster as cluster
from concurrent.futures import ThreadPoolExecutor
from sklearn.pipeline import make_pipeline
from test import best_Kmeans,save_classify_dir
from img import image_local_process

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from img import get_img_paths_by_cluster
from sklearn.preprocessing import StandardScaler
def class_print(top_index,class_idx):
    for i in top_index:
        print(class_idx[i])
def list_classification(top_index,class_idx):
    out_class = []
    for i in top_index:
        out_class.append(class_idx[i])
    return out_class
def predict(img,model,transforms):
    output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1
    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=10)
    final_classify = list_classification(top5_class_indices[0],class_idx)
    return final_classify

if __name__ == '__main__':
    class_idx = label_get()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


    model = timm.create_model("seresnextaa101d_32x8d.sw_in12k_ft_in1k_288",pretrained=True)
    model = model.eval()
    model.reset_classifier(0,'')

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    #
    img_dir = os.path.join(BASE_DIR,"images")

    img_list,imgs = image_local_process(img_dir)
    K = 10
    cluster_model = cluster.KMeans(n_clusters=K)
    #创建一个numpy数组，用于存放所有图片的特征向量
    # 获取实际特征维度
    output_sample = model(transforms(imgs[0]).unsqueeze(0))
    actual_feature_dim = output_sample.flatten().shape[0]

    features = np.zeros((len(imgs),actual_feature_dim))

    #提取flatten后的特征向量
    for i in tqdm(range(len(imgs))):
        output = model(transforms(imgs[i]).unsqueeze(0))  # unsqueeze single image into batch of 1
        flattened = output.detach().flatten().numpy()
        features[i] = flattened

    #数据去重
    # features = np.unique(features,axis=0)
    # scaler = StandardScaler()
    # #todo 标准化
    # features = scaler.fit_transform(features)
    cluster_model.fit(features)

    # best_Kmeans(features)

    # 使用PCA降维到2维，方便可视化
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    labels = cluster_model.predict(features)
    cluster_labels =cluster_model.labels_
    cluster_dict = get_img_paths_by_cluster(cluster_labels, img_list)

    save_classify_dir(cluster_dict)

    for i in range(0,K):
        plt.scatter(reduced_features[labels == i, 0], reduced_features[labels == i, 1],cmap='viridis')
    plt.show()
