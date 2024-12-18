import numpy as np
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram
from ram import get_transform
import torch


pretrained_model = "pretrained/ram_plus_swin_large_14m.pth"
image_size = 384

def generate_image_tags(dataset_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(image_size=image_size)
    model = ram_plus(pretrained=pretrained_model, image_size=image_size, vit='swin_l')
    model.eval()
    model = model.to(device)


    tags_collection = []

    print("\n------------------------------------")
    print("Generating tags for AMBER dataset")
    print("\n------------------------------------")
    for i in range(1, 1005):
        img_path =  f"{dataset_path}/AMBER_{i}.jpg"
        print(img_path)

        image = transform(Image.open(img_path)).unsqueeze(0).to(device)
        res = inference_ram(image, model)

        tags = {"id": i, "tags": res[0].split(" | ")}
        tags_collection.append(tags)
    
    print("Tags generated successfully")
    return tags_collection
