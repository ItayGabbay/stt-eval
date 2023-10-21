from models.base_model import BaseModel
import torchaudio

class BaseEnhancer(BaseModel):
    def __init__(self, config):
        super.__init__(config)

    def file_type(self) -> str:
        return "wav"
    
    def out_sample_rate(self) -> int:
        raise NotImplementedError
    
    def save_output(self, output_filepath, output) -> None:
        torchaudio.save(output_filepath, output, self.out_sample_rate())