from argparse import ArgumentParser
from jiwer import wer

from utils.adjusted_wer import compute_wer, compute_wer_no_punctuation
from tqdm import tqdm
import pandas as pd
import os

def load_blob_names_for_eval(labeled_data_folder, model_outputs_folder):
    model_outputs_files = os.listdir(model_outputs_folder)
    labeld_data_files = os.listdir(labeled_data_folder)

    model_output_names = set(model_outputs_files)
    labeld_data_names = set(labeld_data_files)
    intersection = model_output_names.intersection(labeld_data_names)

    filtered_model_outputs_blobs = sorted([file_name for file_name in model_outputs_files if file_name in intersection])
    filtered_labeld_data_blobs = sorted([file_name for file_name in labeld_data_files if file_name in intersection])

    assert len(filtered_model_outputs_blobs) == len(filtered_labeld_data_blobs)

    print(f"Found {len(filtered_model_outputs_blobs)} files to evaluate out of {len(model_outputs_files)} model outputs and {len(labeld_data_files)} labeld data files")

    return filtered_model_outputs_blobs, filtered_labeld_data_blobs

def eval(args):
    model_outputs_blobs, labeld_data_blobs = load_blob_names_for_eval(args.labeld_data_folder, args.model_outputs_folder)

    total_regular_wer = 0
    total_clean_adjusted_wer = 0
    total_adjusted_wer = 0
    
    results_data = []

    for model_file_name, labeled_file_name in tqdm(zip(model_outputs_blobs, labeld_data_blobs)):
        if model_file_name != labeled_file_name:
            raise ValueError("model output file and labeld data file have different name")

        # read files
        with open(os.path.join(args.model_outputs_folder, model_file_name), 'r', encoding='utf-8') as f:
            model_output_text = f.read()
        with open(os.path.join(args.labeld_data_folder, labeled_file_name), 'r', encoding='utf-8-sig') as f:
            labeld_data_text = f.read()

        model_output_text = model_output_text.strip()
        labeld_data_text = labeld_data_text.strip()

        regular_wer = wer(labeld_data_text, model_output_text)
        total_regular_wer += regular_wer * 100

        clean_adjusted_wer = compute_wer_no_punctuation(labeld_data_text, model_output_text)['wer']
        total_clean_adjusted_wer += clean_adjusted_wer

        adjusted_wer = compute_wer(labeld_data_text, model_output_text)['wer']
        total_adjusted_wer += adjusted_wer

        result_data = {
            "file_name": model_file_name,
            "prediction": model_output_text,
            "label": labeld_data_text,
            "prediction_len": len(model_output_text.split(' ')),
            "label_len": len(labeld_data_text.split(' ')),
            "regular_wer": regular_wer * 100,
            "adjusted_wer": adjusted_wer,
            "clean_adjusted_wer": clean_adjusted_wer
        }

        results_data.append(result_data)

    mean_regular_wer = total_regular_wer / len(model_outputs_blobs)
    mean_clean_adjusted_wer = total_clean_adjusted_wer / len(model_outputs_blobs)
    mean_adjusted_wer = total_adjusted_wer / len(model_outputs_blobs)


    # save results_data to csv
    df = pd.DataFrame(results_data)

    df.to_csv(os.path.join(args.eval_output_path, "eval_report.csv"))
    
    # create new file in results path, and write the wer score. file name is the model name
    results = f"mean_regular_wer: {str(mean_regular_wer)}\nmean_adjusted_clean_wer: {str(mean_clean_adjusted_wer)}\nmean_adjusted_wer: {str(mean_adjusted_wer)}"
    print(results)
    
    # save restuls to txt file
    with open(os.path.join(args.eval_output_path, "mean_wers.txt"), 'w') as f:
        f.write(results)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--labeld_data_folder') # Path to a folder that contains txt files with true labels. example: data/lables
    parser.add_argument('--model_outputs_folder') # Path to a folder that contains txt files with model outputs. example: data/outputs/whisper_hf
    parser.add_argument('--eval_output_path') # Path to the folder in which to write the output. example: eval_outputs/whisper_hf
    args = parser.parse_args()

    create_if_not_exists = os.makedirs(args.eval_output_path, exist_ok=True)

    eval(args)