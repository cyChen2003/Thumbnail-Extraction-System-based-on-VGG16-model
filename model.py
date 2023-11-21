from urllib.request import urlopen
from PIL import Image
import timm
import torch
import os
from labels import label_get
from torchvision import models
import requests
from io import BytesIO
from img import image_process
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

    img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = image_process(img_url)


    model = timm.create_model("vgg19.tv_in1k",pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    final_classify=predict(img,model,transforms)
    print(final_classify)