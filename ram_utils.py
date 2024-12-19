import numpy as np
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram
from ram import get_transform
import torch


pretrained_model = "pretrained/ram_plus_swin_large_14m.pth"
image_size = 384


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=image_size)
    model = ram_plus(pretrained=pretrained_model, image_size=image_size, vit='swin_l')
    model.eval()
    model = model.to(device)
    return model, transform, device


def generate_tags(image_path, model, transform, device):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)
    res = inference_ram(image, model)
    return res[0].split(" | ")
    
