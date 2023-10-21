import azure.cognitiveservices.speech as speechsdk
from models.stt.BaseSTT import BaseSTT
from vyper import v

v.automatic_env()

class CognitiveServicesStreamToText(BaseSTT):
    def __init__(self, config):
        self.ln = config["ln"]
        # Creates an instance of a speech config with specified subscription key and service region.
        # Replace with your own subscription key and region identifier from here: https://aka.ms/speech/sdkregion
        # Loading parameters from environment variables

        speech_endpoint, speech_key, service_region = v.get("SPEECH_ENDPOINT"), v.get("SPEECH_KEY"), v.get("SPEECH_REGION")
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        # speech_config.endpoint_id ="endpoint"
        speech_config.speech_recognition_language = self.ln  # "EN-us"
        self.speech_config = speech_config

    def infer(self, input_filename):
        audio_config = speechsdk.audio.AudioConfig(filename=input_filename)     # for local files insert the path to the folder

        # Creates a recognizer with the given settings
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
        res = speech_recognizer.recognize_once_async().get()
        
        return res.text

        