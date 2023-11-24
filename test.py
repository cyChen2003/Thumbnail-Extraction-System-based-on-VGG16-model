from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm
def best_Kmeans(X):
    K = range(1, 50)
    distortions = []
    for k in tqdm(K):
        # 分别构建各种K值下的聚类器
        Model = KMeans(n_clusters=k).fit(X)
        # 计算各个样本到其所在簇类中心欧式距离(保存到各簇类中心的距离的最小值)
        distortions.append(sum(np.min(cdist(X, Model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # 绘制各个K值对应的簇内平方总和，即代价函数SSE
    # 可以看出当K=3时，出现了“肘部”，即最佳的K值。
    plt.plot(K, distortions, 'bx-')
    # 设置坐标名称
    plt.xlabel('optimal K')
    plt.ylabel('SSE')
    plt.show()
def save_classify_dir(cluster_dict):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 创建cluster_output文件夹
    dir_path = os.path.join(BASE_DIR,"cluster_output")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        # 清空cluster_output文件夹
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    for cluster_labels, img_paths in cluster_dict.items():
        # 创建各个类别文件夹
        cluster_dir_path = os.path.join(dir_path,str(cluster_labels))
        if not os.path.exists(cluster_dir_path):
            os.makedirs(cluster_dir_path)
        else:
            # 清空cluster_output文件夹
            shutil.rmtree(cluster_dir_path)
            os.makedirs(cluster_dir_path)
        # 将图片复制到对应的类别文件夹中
        for img_path in img_paths:
            shutil.copy(img_path,cluster_dir_path)
        # 将图片复制到对应的类别文件夹中
