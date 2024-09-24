import os
import pandas as pd
import argparse
from collections import Counter

def count_images_per_class(train_dir, output_csv=None):
    class_image_count = Counter()

    # Explore through folder list
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            # Count the number of files in class that start with 'sketch'
            image_count = sum(1 for file in os.listdir(class_path) if file.lower().startswith('sketch') and file.lower().endswith(('.png', '.jpg', '.jpeg')))
            class_image_count[class_name] = image_count
            print(f"Class {class_name}: {image_count} images")

    # Save to csv if output_csv is provided
    if output_csv:
        df = pd.DataFrame(list(class_image_count.items()), columns=['class_name', 'image_count'])
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    return class_image_count

def main():
    # Set argument parser
    parser = argparse.ArgumentParser(description="Count and visualize number of images per class in the train dataset.")
    parser.add_argument('--train_dir', required=True, help="Path to the merged train dataset directory.")
    parser.add_argument('--output_csv', help="Path to the output CSV file to save image counts per class.")
    args = parser.parse_args()

    # Activate function
    result = count_images_per_class(args.train_dir, args.output_csv, args.visualize)
    print(f"Total classes: {len(result)}")
    print(f"Total images: {sum(result.values())}")

if __name__ == "__main__":
    main()