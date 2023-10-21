import os
from argparse import ArgumentParser
import re

def build_lexicon(united_labels_input, output_path):
    pattern = re.compile("[A-Za-z0-9]+")

    all_words = set()
    with open(united_labels_input, "r", encoding="utf-8-sig") as f:
        lines = f.read().split('\n')

    for line in lines:
        line_words = line.split(' ')
        all_words.update(line_words)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        for i, word in enumerate(all_words):
            spelling = " ".join([c for c in word])
            # if pattern.fullmatch(word) is not None:
            #     to_write = f"{word} {spelling} |"
            # else:
            #     to_write = f"{spelling} {word} |"
            
            to_write = f"{word} {spelling} |"
            



            if i != len(all_words) - 1:
                to_write +="\n"

            f.write(to_write)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--united_labels_input', default="data/united_train_labels_for_ngram/united.txt")
    parser.add_argument('--out_path', default="models/ctc_lm/lexicon.txt")
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        os.remove(args.out_path)


    build_lexicon(args.united_labels_input, args.out_path)