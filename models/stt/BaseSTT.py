from models.base_model import BaseModel

class BaseSTT(BaseModel):
    def __init__(self, config):
        super.__init__(config)

    def file_type(self) -> str:
        return "txt"
    
    def save_output(self, output_filepath, transcript) -> None:
        print(f"Transcript: {transcript}")
            
        # write transcript to file
        with open(output_filepath, 'w') as f:
            f.write(transcript)