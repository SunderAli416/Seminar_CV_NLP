# Image Captioning

### Vision and Language - WS 24/25

#### Project Members:
- Sundar Ali
- Aman Soofi
- Muhammad Salman Naseem
- Muhammad Qamar Amin

## Abstract
In this project, we performed different experiments for image captioning through vision and language model fusion within a pipeline specifically designed to reduce hallucinations in multimodal AI. We leverage a state-of-the-art vision captioner, optical character recognition (OCR) object detection, and image tagging modules, combined with a large language model (LLM), to generate captions that are both comprehensive and grounded in the image content.

## Technology Stack
- Python 3
- Florence 2
- Llama 3.2
- RAM++
- Owl-ViT-v2
- Paddle-OCR
- AMBER Dataset

## Setup
### AMBER Dataset
 - Download [ AMBER Dataset](https://drive.google.com/file/d1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing)
 - Create a folder 'data' in root directory 
 - Extract dataset in 'data' (data/image/AMBER_x.jpg).

### RAM++
 - Create a folder "pretrained" in root directory 
 - Download [checkpoint file](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) in this directory
 - Install RAM++ as package
 `pip install git+https://github.com/xinyu1205/recognize-anything.git`
 
## Inference
Each expriment has its own notebook file. Run any experiment and it will create a json output file. Use this file to run inference.

`python inference.py --inference_data path/to/your/inference/file --evaluation_type g`