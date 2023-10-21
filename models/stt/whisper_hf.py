from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import torchaudio.functional as F
import torch
from models.stt.BaseSTT import BaseSTT

class WhisperHuggingFace(BaseSTT):
    def __init__(self, config) -> None:
        self.config = config

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"using device: {self.device}")

        # load model and processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to(self.device)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=config['ln'], task="transcribe")

    def infer(self, input_filename):
        # audio and sr from file 
        wavs, sr = sf.read(input_filename)
        resampled_waveform = F.resample(torch.tensor(wavs), sr, 16000)

        input_features = self.processor(resampled_waveform, sampling_rate=16000, return_tensors="pt").input_features

        # generate token ids
        predicted_ids = self.model.generate(input_features.to(self.device), forced_decoder_ids=self.forced_decoder_ids)
        
        # decode token ids to text
        # transcription = self.processor.batch_decode(predicted_ids)
        # ['<|startoftranscript|><|fr|><|transcribe|><|notimestamps|> Un vrai travail intéressant va enfin être mené sur ce sujet.<|endoftext|>']

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription[0]

if __name__ == "__main__":

    file = "/eyesondatadisk/Dor/tts_fun/temp/audio_847639@02-03-2020_00-02-11_0.wav"
    
    # wavs, sr = sf.read(file)
    # resampled_waveform = F.resample(torch.tensor(wavs), sr, 16000)
    # print(resampled_waveform.shape)

    whisper = WhisperHuggingFace({})
    print('get from file')
    txt = whisper.get_text_from_file(file)
    print('end')
    print(txt)