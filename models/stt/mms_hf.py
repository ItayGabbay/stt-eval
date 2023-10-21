import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
import soundfile as sf
from scipy.signal import resample
from models.stt.BaseSTT import BaseSTT
from models.ctc_lm.beam_search_lm_decoder import beam_ngram_decoder


class MMSStreamToText(BaseSTT):
    def __init__(self, config) -> None:
        self.ln = config['ln']
        self.expected_rate = 16000
        
        # model_id = "facebook/mms-1b-all"
        model_path = "../../roye/tts_fun/models/mms-1b-all-adapters-cont-3/checkpoint-7911/"
        vocab_path = "../../roye/tts_fun/"

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", target_lang="ara")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_path,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
            ignore_mismatched_sizes=True,
        ).to("cuda:1")
        self.model.config.ctc_zero_infinity = True


        tokens = [item[1] for item in sorted(self.processor.tokenizer.decoder.items())]

        blank_token = self.processor.tokenizer.pad_token
        sil_token = self.processor.tokenizer.word_delimiter_token
        unk_word = self.processor.tokenizer.unk_token
        self.decoder = beam_ngram_decoder(tokens=tokens, n_gram=5, blank_token=blank_token, sil_token=sil_token, unk_word=unk_word)
    
    def infer(self, input_filename):
        data, original_sample_rate = sf.read(input_filename)
        data_resampled = resample(data, int(len(data) * self.expected_rate / original_sample_rate), axis=0)
        inputs = self.processor(data_resampled, sampling_rate=self.expected_rate, return_tensors="pt").to("cuda:1")

        with torch.no_grad():
            outputs = self.model(**inputs).logits

        beam_result = self.decoder(outputs.cpu())

        # tokens_str = "".join(self.decoder.idxs_to_tokens(beam_result[0][0].tokens))
        # decoder_transcripts = " ".join(tokens_str.split("|")).strip()

        decoder_transcripts = " ".join(beam_result[0][0].words).strip()

        return decoder_transcripts

        # ids = torch.argmax(outputs, dim=-1)[0]
        # transcription = self.processor.decode(ids)
        # return transcription