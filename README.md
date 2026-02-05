# 🧾 Synthetic Persian OCR Data Generator

A high-quality **synthetic image–text dataset generator** for **Persian / mixed Persian–English OCR and Vision-Language Models (VLMs)**.

Generates realistic document-like images with:

- Persian text  
- Mixed Persian–English tokens  
- Quantities, units, numbering  
- Random fonts, backgrounds, colors, distortions, rotations  

Designed for:

- Vision-Language Model training
- Low-resource Persian text recognition  
- Robustness to layout, font, and background variations  



## ✨ Features

-  **Persian-first OCR generation**  
-  Mixed Persian / English / numeric text rendering  
-  Random fonts, colors, and backgrounds  
-  Page-like layouts with numbering and lists  
-  Geometric distortions (conformal warp)  
-  Realistic image preprocessing (CLAHE, sharpening, resizing)  
-  Random rotations (0°, ±90°, 180°)  
-  PyTorch `Dataset` + `Collator` ready  
-  Compatible with **Dots.OCR style training**  



## 📁 Project Structure

```text
├── main.py                 # Dataset generation / visualization script
├── data_loader.py          # PyTorch Dataset + Collator
├── requirements.txt
├── backgrounds/            # Background images (jpg/png/webp)
├── fonts/
│   ├── persian_fonts/      # Persian fonts (.ttf/.otf)
│   └── english_fonts/      # English fonts
├── generated_data/
│   ├── images/
│   └── labels/
└── README.md
```


⚙️ Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/imaryamsamani-lang/Image-Data-Generator.git
cd Image-Data-Generator
pip install -r requirements.txt
```

Download the fonts and extract the into the fonts folder.

persian fonts:
https://drive.google.com/file/d/18JBa3f-4_tw2MgDiW6Po_-ydDdW11_8S/view?usp=drive_link

english fonts:
https://drive.google.com/file/d/1YoSQN6qhEtqpdI-x6ONxn7EAUW70fJ8q/view?usp=sharing

Add background images to the backgrounds folder. Some samples are provided here: 
https://drive.google.com/file/d/1GsIPEeqV_rzKYY6nsR-Rcxj9vz9AB1V2/view?usp=sharing


📊 Input Data

The generator expects a CSV file with a text column, We have used the Dehkhoda dataset for it:

```text

0 بسم الله الرحمن الرحيم

1	کتابخانه

2	اسلامی المعارف بنسیاد دایرة

3	روه ادبات

4	به قلم گروهی از نویسندگان

... ...
3704188	پایان
```

dehkhoda.csv is available at:
https://drive.google.com/file/d/1mxMMTlPqATtShRoDYpgJdfcDarQIEjir/view?usp=drive_link

## Usage

1. Generate and save synthetic data

```bash
python main.py --save --output_path generated_data --max_samples 1000
```

This will produce:

```bash
generated_data/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── labels/
    ├── 0.txt
    ├── 1.txt
    └── ...
```

Each image has a corresponding UTF-8 Persian label.

2. Visualize samples (debug mode, training format)
```bash
python main.py --visualize
```

## Dataset Output Format

Each dataset item returns a dictionary:

```python
{
  "image": PIL.Image,
  "answer": str,        # raw Persian text
  "prompt_only": str,
  "text_full": str
}
```

This format is directly compatible with ision-Language fine-tuning.

## Output Results

![Alt text](p/home/maryam/Desktop/Synthetic_Data_Generator/generated_data/images/1.png)

## Training Integration (Dots.OCR example)

The included Collator:

Handles vision inputs via process_vision_info

Masks prompt tokens correctly

Supports multi-token <|assistant|> markers

Produces labels for causal LM training

## ⚠️ Important Notes

Do NOT reshape Persian text when saving labels — arabic_reshaper is only for visualization

Fonts must support Persian glyphs

Background images should be high resolution

This is a synthetic generator, not a real OCR dataset
