import requests
import os
from PIL import Image
from io import BytesIO
def image_process(img_url):
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
    }
    response = requests.get(img_url, headers=headers, stream=True, verify=False, timeout=20,proxies={'http':'http://localhost:7890','https':'http://localhost:7890'})
    image_data = response.content
    image_file = BytesIO(image_data)
    img = Image.open(image_file).convert('RGB')
    return img