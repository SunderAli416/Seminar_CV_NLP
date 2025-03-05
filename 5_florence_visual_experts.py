import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/image" , help="Path to image dataset")
    parser.add_argument("--tag_run", action='store_true', help="Flag to run image tagging")
    parser.add_argument("--tag_file", type=str, default='results/tags.json', help="Path to the existing tags file")
    parser.add_argument("--det_run", action='store_true', help="Flag to run object detection")
    parser.add_argument("--det_file", type=str, default='results/det.json', help="Path to the existing detection file")
    parser.add_argument("--cap_run", action='store_true', help="Flag to run initial caption generation")
    parser.add_argument("--cap_file", type=str, default='results/cap.json', help="Path to existing initial caption")
    args = parser.parse_args()
    return args

# Method to save the data to a file
def save_file(data, file):
    print("------------------------------------")
    print(f"Saving data to {file}")
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
    print("Data saved successfully")


# Method to load tags from the existing tags file
def load_file(file):
    print("------------------------------------")
    print(f"Loading data from {file}")
    data = []
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
    print("Data loaded successfully")
    return data


# Method to generate tags for the images in the dataset
def generate_image_tags(dataset_path, tag_file):
    print("------------------------------------")
    print("Generating tags for AMBER dataset")
    print("------------------------------------")

    import ram_utils as ram

    tags_collection = []
    model, transform, device = ram.load_model()

    for i in range(1, 1005):
        image_path =  f"{dataset_path}/AMBER_{i}.jpg"
        print(image_path)

        tags = ram.generate_tags(image_path=image_path, model=model, transform=transform, device=device)
        tags_collection.append({"id": i, "tags": tags})
    
    print("Tags generated successfully")    
    save_file(data=tags_collection, tag_file=tag_file)

    return tags_collection


# Method to detect objects in the images based in tags
def detect_objects(dataset_path, tags_collection, det_file):
    print("------------------------------------")
    print("Detecting Objects for AMBER dataset")
    print("------------------------------------")

    import owl_utils as owt

    object_collection = []
    processor, model = owt.load_model()

    for tag in tags_collection:
        try:
            image_path =  f"{dataset_path}/AMBER_{tag['id']}.jpg"

            if (len(tag['tags']) == 0):
                print(f"No tags found for image {tag['id']}")
            else:
                print(image_path)
                detections = owt.detect_objects(image_path=image_path, texts=tag['tags'], processor=processor, model=model)
                object_collection.append({"id": tag['id'], "detections": detections})
        except Exception as e:
            print(f"Error processing image {tag['id']}: {e}")
    
    print("Object Detection Complete")
    print(object_collection)
    save_file(data=object_collection, file=det_file)

    return tags_collection

def generate_initial_caption(dataset_path, cap_file):
    print("------------------------------------")
    print("Using Florance for AMBER dataset")
    print("------------------------------------")

    from florence_utils import run_inference

    initial_captions = []
    for i in range(1, 1005):
        try:
            image_path =  f"{dataset_path}/AMBER_{i}.jpg"
            print(image_path)

            object_detection_info = run_inference(image_path, '<OD>')
            ocr_info = run_inference(image_path, '<OCR>')
            caption = run_inference(image_path, '<CAPTION>')

            response = {"id": i, "caption": caption, "object_detection_info": object_detection_info, "ocr_info": ocr_info}
            initial_captions.append(response)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    
    print("Florance Run Complete")
    print(initial_captions)
    save_file(data=initial_captions, file=cap_file)
    return initial_captions
    
def generate_final_caption(initial_captions ,result_file, tags, detections):
    print("------------------------------------")
    print("Using LLAMA")
    print("------------------------------------")
    from llama_utils import call_llama, call_llama_with_extra_data_exp5

    caption_list = []
    for i in range(1, 1005):
        try:
            ocr_info = initial_captions[i-1]['ocr_info']
            object_detection_info = initial_captions[i-1]['object_detection_info']
            caption = initial_captions[i-1]['caption']
            extra_tags = tags[i-1]['tags']
            extra_detections = detections[i-1]['detections']

            # response = call_llama(ocr_info, object_detection_info,caption)
            # print(f"1: Response for image {i}: {response}")
            
            response_extra = call_llama_with_extra_data_exp5(ocr_info, object_detection_info,caption, extra_tags, extra_detections)
            caption = {"id": i, "response": response_extra}
            print(caption)
            caption_list.append(caption)
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    
    print("LLAMA Run Complete")
    print(caption_list)
    save_file(data=caption_list, file=result_file)
    return caption_list

def main(args):

    tags_collection = []
    object_collection = []
    if args.tag_run or not os.path.exists(args.tag_file):
        tags_collection = generate_image_tags(
            dataset_path = args.image,
            tag_file = args.tag_file)
    else:
        tags_collection = load_file(file=args.tag_file)
    
    if len(tags_collection) != 0:
        if args.det_run or not os.path.exists(args.det_file):
            object_collection = detect_objects(
                dataset_path = args.image,
                tags_collection = tags_collection,
                det_file = args.det_file)
        else:
            object_collection = load_file(file=args.det_file)
    
    if args.cap_run or not os.path.exists(args.cap_file):
        initial_captions = generate_initial_caption(dataset_path = args.image, cap_file = args.cap_file)
    else:
        initial_captions = load_file(file=args.cap_file)
    
    if(len(initial_captions) != 0):
        generate_final_caption(initial_captions, "generated_captions/5_florence_visual_experts.json", tags_collection, object_collection)
    

if __name__ == "__main__":
    args = get_args()
    main(args)