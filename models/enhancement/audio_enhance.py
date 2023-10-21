import torch
import torchaudio
from speechbrain.pretrained import SepformerSeparation, SpectralMaskEnhancement
from vyper import v
from models.enhancement.base_enhancer import BaseEnhancer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SepformerDns4SpeechEnhancer(BaseEnhancer):
    def __init__(self, config) -> None:
        self.device = DEVICE
        self.source = "speechbrain/sepformer-whamr-enhancement"
        self.savedir = "pretrained_models/sepformer-dns4-enhancement"
        self.enhance_model = SepformerSeparation.from_hparams(source=self.source, savedir=self.savedir, run_opts={'device': self.device})
        print("Using device:", DEVICE)

    def out_sample_rate(self) -> int:
        return 8000

    def infer(self, input_filename) -> torch.Tensor:
        # for custom file, change path
        est_sources = self.enhance_model.separate_file(path=input_filename.replace('\\', '/'))
        # saving to tmp file in order to upload to blob storage
        return est_sources[:, :, 0].detach().cpu()



class MetriganPlusSpeechEnhancer(BaseEnhancer):
    def __init__(self, config) -> None:
        self.device = DEVICE
        self.source = "speechbrain/metricgan-plus-voicebank"
        self.savedir = "pretrained_models/metricgan-plus-voicebank"
        # self.enhance_model = SpectralMaskEnhancement.from_hparams(source=self.source, savedir=self.savedir, run_opts={'device': self.device})
        print("Using device:", self.device)
    
    def out_sample_rate(self) -> int:
        return 16000

    def infer(self, input_filename) -> torch.Tensor:
        # Load and add fake batch dimension
        noisy = self.enhance_model.load_audio(input_filename.replace('\\', '/')).unsqueeze(0).to(self.device)

        # Add relative length tensor
        enhanced = self.enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]).to(self.device))
        return enhanced.cpu()

    