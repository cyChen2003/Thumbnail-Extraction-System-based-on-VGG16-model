import requests
import os
from PIL import Image
from io import BytesIO
import glob
import cairosvg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import aspose.words as aw
def svg_to_jpg(svg):
    try:
        doc = aw.Document()
        jpg = svg.replace(".svg","")
        builder = aw.DocumentBuilder(doc)
        shape = builder.insert_image(svg)
        shape.image_data.save(jpg + ".jpg")
    except Exception as e:
        print(f"Error: {svg}\n{e}")
def jpg_to_png(jpg):
    try:
        img = Image.open(jpg)
        img.save(jpg.replace(".jpg",".png"))
    except Exception as e:
        print(f"Error: {jpg}\n{e}")
def image_process(img_url):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
    }
    response = requests.get(img_url, headers=headers, stream=True, verify=False, timeout=20,proxies={'http':'http://localhost:7890','https':'http://localhost:7890'})
    image_data = response.content
    image_file = BytesIO(image_data)
    img = Image.open(image_file).convert('RGB')
    return img
def image_local_process(dir_path):
    img_list = glob.glob(dir_path + "/*")
    img_list.sort()
    img = []

    for i in img_list:
        if i.endswith(".jpg") or i.endswith(".png") or i.endswith(".jpeg"):
            try:
                img.append(Image.open(i).convert('RGB'))
            except Exception as e:
                print(f"Error: {i}\n{e}")
        elif i.endswith(".svg"):
            try:
                svg_to_jpg(i)
                img.append(Image.open(i.replace(".svg",".jpg")).convert('RGB'))
                #删除svg文件
                os.remove(i)
            except Exception as e:
                print(f"Error: {i}\n{e}")

    return img_list, img
def get_img_paths_by_cluster(labels, img_paths):
    cluster_dict = {}
    for label, img_path in zip(labels, img_paths):
        if label not in cluster_dict:
            cluster_dict[label] = [img_path]
        else:
            cluster_dict[label].append(img_path)
    return cluster_dict
def all_picture_to_png(dir_path):
    img_list = glob.glob(dir_path + "/*")
    img_list.sort()
    for i in img_list:
        if i.endswith(".jpg") or i.endswith(".png") or i.endswith(".jpeg"):
            try:
                img = Image.open(i).convert('RGB')
                img.save(i.replace(".jpg",".png"))
                print(i)
            except Exception as e:
                print(f"Error: {i}\n{e}")
        elif i.endswith(".svg"):
            try:
                svg_to_jpg(i)
                jpg_to_png(i.replace(".svg",".jpg"))
                os.remove(i)
            except Exception as e:
                print(f"Error: {i}\n{e}")
    img_list = glob.glob(dir_path + "/*")
    img_list.sort()
    for i in img_list:
        #i结尾不为.png的文件删除
        if not i.endswith(".png"):
            try:
                os.remove(i)
            except Exception as e:
                print(f"Error: {i}\n{e}")