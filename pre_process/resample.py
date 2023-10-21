import os
from argparse import ArgumentParser
import soundfile as sf
from scipy.signal import resample
from tqdm import tqdm

def resample_dir(input_dir, output_dir, target_sr):
    for audio_file in tqdm(os.listdir(input_dir)):
        audio_data, origin_sr = sf.read(os.path.join(input_dir, audio_file))

        if origin_sr != target_sr:
            print(f"resampling {audio_file} from {origin_sr} to {target_sr}")
            audio_data = resample(audio_data, int(len(audio_data) * target_sr / origin_sr), axis=0)

        sf.write(os.path.join(output_dir, audio_file), audio_data, target_sr)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--wavs_input', default="/mnt/nfs/stt_project/dataset/reupload/test/")
    parser.add_argument('--wavs_output', default="/mnt/nfs/dorten/dataset/test_wav")
    parser.add_argument('--target_sr', default=16000)
    args = parser.parse_args()

    out_path = f"{args.wavs_output}_{args.target_sr}"

    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(out_path):
        os.remove(os.path.join(args.out_path, file))


    resample_dir(args.wavs_input, out_path, args.target_sr)