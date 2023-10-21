import os
from argparse import ArgumentParser
from utils.clean_arabic import clean_arabic


def is_contain_english(text):
    return "name" in text.lower()

def to_ignore(text:str) -> bool:
    if "unknown" in text.lower():
        return True
    elif "*" in text:
        return True
    elif is_contain_english(text):
        return True
    else:
        return False
    

def clean_text(text:str) -> str:
    text = clean_arabic(text)
    cleaned_string = text.replace("[", "").replace("]", "")
    return cleaned_string


def clean_labels(original_labels_path, output_path):
    labels = []
    for i, label_file in enumerate(os.listdir(original_labels_path)):
        with open(os.path.join(original_labels_path, label_file), "r", encoding="utf-8-sig") as f:
            text = f.read().strip()
            
        if to_ignore(text):
            continue

        cleaned = clean_text(text)

        with open(os.path.join(output_path, label_file), "w", encoding="utf-8-sig") as f:
            f.write(cleaned)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--labels_input', default="/mnt/nfs/stt_project/dataset/train-txt")
    parser.add_argument('--out_path', default="/mnt/nfs/dorten/cleaned_labels/train")
    args = parser.parse_args()

    for file in os.listdir(args.out_path):
        os.remove(os.path.join(args.out_path, file))


    clean_labels(args.labels_input, args.out_path)