import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/image" , help="Path to image dataset")
    parser.add_argument("--tag_run", action='store_true', help="Flag to run image tagging")
    parser.add_argument("--tag_file", type=str, default='data/tags.json', help="Path to the existing tags file")
    parser.add_argument("--det_run", action='store_true', help="Flag to run object detection")
    parser.add_argument("--det_file", type=str, default='data/det.json', help="Path to the existing detection file")
    args = parser.parse_args()
    return args

output_data = []

# Loop through each image file and perform the inference for each task


    # Generate Image Tags
    # generate_image_tags(img_path)

#     # Run inference with each task type
#     object_detection_info = run_inference(img_path, '<OD>')
#     ocr_info = run_inference(img_path, '<OCR>')
#     detailed_caption = run_inference(img_path, '<CAPTION>')
    
#     # Generate final response using call_llama
#     # base64_image=convert_to_base64(img_path)
#     # final_response = call_llama(ocr_info, object_detection_info,base64_image)
#     final_response = call_llama(ocr_info, object_detection_info,detailed_caption)
#     # basic_response=call_llama_basic(base64_image)
#     print(f"Response for image {i}: {final_response}")
#     # print(f"Basic Response for image {i}: {basic_response}")
#     # Append the result for this image
#     output_data.append({"id": i, "response": final_response})

# # Save the results to a JSON file
# output_file = "evaluation_results/amber_evaluation_results.json"
# with open(output_file, "w") as file:
#     json.dump(output_data, file, indent=4)

# print(f"Evaluation results saved to {output_file}")


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

if __name__ == "__main__":
    args = get_args()
    main(args)