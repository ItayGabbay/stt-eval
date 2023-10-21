from argparse import ArgumentParser
from vyper import v
from models.enhancement.audio_enhance import SepformerDns4SpeechEnhancer, MetriganPlusSpeechEnhancer
from models.stt.cogitive_services_stt import CognitiveServicesStreamToText
from models.stt.whisper_azure_aoi import WhisperStreamToText
from models.stt.mms_hf import MMSStreamToText
from models.stt.whisper_hf import WhisperHuggingFace
from models.stt.BaseSTT import BaseModel
from models.stt.mms_arabic import MMSArabic
import os
from tqdm import tqdm
v.automatic_env()
done = False

def model_factory(model_name, config) -> BaseModel: 
    class_dict = {
        'cs-speech': CognitiveServicesStreamToText,
        'whisper_api': WhisperStreamToText,
        'whisper_hf': WhisperHuggingFace,
        'mms': MMSStreamToText,
        'mms_arabic': MMSArabic,
        'sepformer-dns4-8k': SepformerDns4SpeechEnhancer,
        'metrigan-plus-16k': MetriganPlusSpeechEnhancer
    }

    if model_name not in class_dict:
        raise Exception(f"Unknown model {model_name}")
    
    return class_dict[model_name](config)


def infer(model_name, model_config, input_wavs_folder_path, output_folder_path, skip_exsists, max_iter=-1):
    # Init API class
    model = model_factory(model_name, model_config)

    for i, input_wav_filename in tqdm(enumerate(os.listdir(input_wavs_folder_path))):
        if 0 < max_iter <= i:
            break

        output_filename = input_wav_filename.replace('.wav', f'.{model.file_type()}')
        output_full_path = os.path.join(output_folder_path, output_filename)

        if skip_exsists and os.path.exists(output_full_path):
            print(f"Skipping {output_full_path} as it already exists")
            continue

        print(f"Processing file: {input_wav_filename}")

        input_full_path = os.path.join(input_wavs_folder_path, input_wav_filename)
        output = model.infer(input_full_path)
        model.save_output(output_full_path, output)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_name', required=True) # the name of the model. used to create the right model
    parser.add_argument('--language', required=True) # the language to inject the model. some models use it
    parser.add_argument('--input_wavs_folder_path', required=True) # the path to the folder that contains the wavs to infer. for example, data/wavs
    parser.add_argument('--output_folder_path', required=True) # the path to the folder that will contain the outputs. for example, data/outputs/metricgan or data/outputs/MMS
    parser.add_argument('--skip_exists', default=True)
    parser.add_argument('--max_iter', default=-1, type=int)
    args = parser.parse_args()

    model_config = {'ln': args.language}

    os.makedirs(args.output_folder_path, exist_ok=True)
    
    infer(args.model_name, model_config, args.input_wavs_folder_path, args.output_folder_path, args.skip_exists, max_iter=args.max_iter)
