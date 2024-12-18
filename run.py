import argparse
import json
import os
from ram_utils import generate_image_tags

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/image" , help="Path to image dataset")
    parser.add_argument("--tag_run", action='store_true', help="Flag to run image tagging")
    parser.add_argument("--tag_file", type=str, default='data/tags.json', help="Path to the existing tags file")
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

def save_image_tags(tags, tag_file):
    print("\n------------------------------------")
    print(f"Saving tags to {tag_file}")
    with open(tag_file, 'w') as f:
        json.dump(tags, f, indent=4)
    print("Tags saved successfully")

def load_image_tags(tag_file):
    print("\n------------------------------------")
    print(f"Loading tags from {tag_file}")
    return []


def main(args):
    if args.tag_run:
        tags = generate_image_tags(dataset_path=args.image)
        save_image_tags(tags, tag_file=args.tag_file)
    else:
        tags = load_image_tags(tag_file=args.tag_file)

if __name__ == "__main__":
    args = get_args()
    main(args)