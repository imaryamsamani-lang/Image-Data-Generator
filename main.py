import argparse
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Subset
from data_loader import DotsOcrJsonl, Collator

import arabic_reshaper
from bidi.algorithm import get_display

def fix_persian_text(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def parse_args():

    parser = argparse.ArgumentParser(description="Generate OCR image-text dataset")

    parser.add_argument("--csv_path", type=str, default="dehkhoda.csv", help="Path to Dehkhoda CSV file")
    
    parser.add_argument("--output_path", type=str, default="generated_data/", help="Output directory")

    parser.add_argument("--save", action="store_true", default=True, help="Save images and labels")

    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize samples")

    parser.add_argument("--max_samples", type=int, default=10, help="Number of samples to process")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_path, encoding="utf-8")
    df["text"] = df["text"].str.replace(r"[()\[\]|]", "", regex=True)

    print("Len of Dehkhoda text:", len(df))

    # Placeholder processor (load later)
    processor = ""

    # Dataset
    data_train = DotsOcrJsonl(df, processor, "train")
    train_ds = Subset(data_train, list(range(len(data_train))))

    collate = Collator(processor)

    indices = list(range(len(train_ds)))
    random.shuffle(indices)

    # Visualization
    if args.visualize:
        for i, ind in enumerate(indices[:args.max_samples]):
            item = train_ds[ind]

            plt.imshow(np.array(item["image"]))
            plt.axis("off")
            plt.show()

            print(fix_persian_text(item["answer"]))
            print("_" * 150)

    # Save
    if args.save:
        images_dir = os.path.join(args.output_path, "images")
        labels_dir = os.path.join(args.output_path, "labels")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for i, ind in enumerate(indices[:args.max_samples]):
            item = train_ds[ind]

            plt.imsave(os.path.join(images_dir, f"{i}.png"), np.array(item["image"]))

            with open(os.path.join(labels_dir, f"{i}.txt"), "w", encoding="utf-8") as f:
                f.write(item["answer"])


if __name__ == "__main__":
    main()


