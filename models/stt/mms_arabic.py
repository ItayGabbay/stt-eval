import torch
import soundfile as sf
from scipy.signal import resample
from models.stt.BaseSTT import BaseSTT
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from lang_trans.arabic import buckwalter

resamplers = {
    8000: torchaudio.transforms.Resample(8000, 16000),
    16000: torchaudio.transforms.Resample(16000, 16000)
}

class MMSArabic(BaseSTT):
    def __init__(self, config) -> None:        
        self.processor = Wav2Vec2Processor.from_pretrained("elgeish/wav2vec2-large-xlsr-53-arabic")
        self.model = Wav2Vec2ForCTC.from_pretrained("elgeish/wav2vec2-large-xlsr-53-arabic").eval()

    
    def infer(self, input_filename):
        input, sr = torchaudio.load(input_filename)
        if sr != 16000:
            torchaudio.save(f"temp/{sr}_{input_filename.split('/')[-1]}", input, sr)
        input = resamplers[sr](input).squeeze().numpy()
        inputs = self.processor(input, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            predicted = torch.argmax(self.model(inputs.input_values).logits, dim=-1)
        predicted[predicted == -100] = self.processor.tokenizer.pad_token_id  # see fine-tuning script
        text = self.processor.tokenizer.batch_decode(predicted)[0]
        text = buckwalter.untrans(text)
        return text