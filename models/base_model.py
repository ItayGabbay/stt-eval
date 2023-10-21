class BaseModel:
    def __init__(self, config):
        self.config = config

    def infer(self, input_file_path) -> str:
        raise NotImplementedError
    
    def save_output(self, output_filepath, output) -> None:
        raise NotImplementedError
    
    def file_type(self) -> str:
        raise NotImplementedError