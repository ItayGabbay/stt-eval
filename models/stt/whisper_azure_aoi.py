import requests
from models.stt.BaseSTT import BaseSTT
from vyper import v 
import json 
import time

class WhisperStreamToText(BaseSTT):
    def __init__(self, config) -> None:
        self.ln = config['ln']
        self.key = v.get("WHISPER_KEY")
        self.url = v.get("WHISPER_ENDPOINT")

    def infer(self, file_name):
        payload = {}
        files = [
            ('file', ('audio.wav', open(file_name, 'rb'), 'audio/wav'))
        ]
        headers = {
            'api-key': self.key
        }


        try_count = 0
        sleep = 5
        while True:
            print(f"Already tried {try_count} times")
            response = requests.request("POST", self.url, headers=headers, data=payload, files=files)
            if response.status_code == 429:
                print(f"Too many requests, sleeping {sleep} seconds")
                time.sleep(sleep)
            elif response.status_code == 200:
                print("Success!")
                text = json.loads(response.text)['text']
                break
            else:
                print(f"Error: {response.status_code} sleeping {sleep} seconds")
                raise Exception(f"Error: {response.status_code}")

            try_count += 1
    
        return text