import os
from argparse import ArgumentParser

def unit_labels(original_labels_path, original_wavs_path, output_path):
    labels_files = os.listdir(original_labels_path)
    wavs_files = set([file.split('.')[0] for file in os.listdir(original_wavs_path)])


    for i, label_file in enumerate(labels_files):
        if label_file.split('.')[0] not in wavs_files:
            continue

        with open(os.path.join(original_labels_path, label_file), "r", encoding="utf-8-sig") as f:
            text = f.read().strip()

        with open(os.path.join(output_path, "united.txt"), "a", encoding="utf-8-sig") as f:
            if i == len(labels_files) - 1:
                f.write(f"{text}")
            else:
                f.write(f"{text}\n")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--labels_input', default="/mnt/nfs/dorten/cleaned_labels/train")
    parser.add_argument('--wavs_input', default="/mnt/nfs/stt_project/dataset/reupload/train")
    parser.add_argument('--out_path', default="data/united_train_labels_for_ngram")
    args = parser.parse_args()

    for file in os.listdir(args.out_path):
        os.remove(os.path.join(args.out_path, file))


    unit_labels(args.labels_input, args.wavs_input, args.out_path)