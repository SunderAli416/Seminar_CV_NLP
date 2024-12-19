# OWL-ViT-v2: Object-Wise Localization by Vision Transformers

from PIL import Image
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection

def load_model():
    processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    return processor, model


def detect_objects(image_path, texts, processor, model):
    image = Image.open(image_path)
    inputs = processor(text=texts, images=image, return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=0.2, target_sizes=target_sizes
    )

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    # text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        # print(f"Detected {texts[label]} with confidence {round(score.item(), 3)} at location {box}")
        detections.append({
            "object": texts[label],
            "score": round(score.item(), 3),
            "location": box,
        })
    
    return detections