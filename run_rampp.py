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