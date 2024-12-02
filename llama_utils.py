from langchain_ollama import OllamaLLM
import base64
from io import BytesIO
from PIL import Image
llm = OllamaLLM(model="llava")

def convert_to_base64(file_path):
    pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def call_llama(OCR_INFO,OBJECT_INFO,image_b64):
    llm_with_image_context = llm.bind(images=[image_b64])
    response = llm_with_image_context.invoke(f"""
    You are a powerful multimodal model and you should generate detailed description of this image using additional
    external information such as:
    OCR_INFORMATION: {OCR_INFO['<OCR>']},
    ---------------------------------------------------
    IMAGE_TAGS: {OBJECT_INFO['<OD>']['labels']},
    ---------------------------------------------------
    OBJECT DETECTION INFORMATION: {OBJECT_INFO['<OD>']},
    ---------------------------------------------------
    This additional information is provided to guide your inference,reduce hallucinations and provide information on objects you miss
    Your job is to utilize all this information to generate one small caption describing the image. The caption can be of 1-5 lines and not much longer
    Your response should not contain any other information aside from the caption
    Only include the caption in your response and nothing else
    Do not include any unwanted intro and any unwanted information in the response aside from the caption
    """)
    return response

def call_llama_basic(image_b64):
    llm_with_image_context = llm.bind(images=[image_b64])
    response = llm_with_image_context.invoke(f"""
    Describe the image in detail and everything present in the image
    """)
    return response