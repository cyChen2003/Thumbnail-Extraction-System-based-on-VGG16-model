from PIL import Image
from torchvision import models, transforms
import urllib
import json
def label_get():
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(LABELS_URL) as url:
        labels = json.loads(url.read().decode())
    return labels
