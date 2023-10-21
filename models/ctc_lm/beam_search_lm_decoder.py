from typing import Any
from torchaudio.models.decoder import ctc_decoder, CTCDecoder

def beam_ngram_decoder(n_gram, 
                    tokens,
                    blank_token,
                    sil_token,
                    unk_word,
                    lm_weight = 2,
                    word_score = 0,
                    beam_size=100,
                    lexicon_path = "models/ctc_lm/lexicon.txt") -> None:
    assert n_gram in [2, 3, 4, 5], "n_gram must be 2 or 3 or 4 or 5"
    n_gram_path = f"models/ctc_lm/{n_gram}gram.arpa"

    decoder = ctc_decoder(
        # lexicon=None,
        lexicon=lexicon_path,
        tokens=tokens,
        lm=n_gram_path,
        nbest=1,
        beam_size=beam_size,
        lm_weight=lm_weight,
        beam_threshold = max(beam_size, 50),
        word_score=word_score,
        blank_token = blank_token,
        sil_token = sil_token,
        unk_word = unk_word
    )

    return decoder



if __name__ == 'main':
    decoder = beam_ngram_decoder(n_gram=2, tokens=['a'])
