"""This script is to use the locally installed Mozilla Deepseech package to convert input audio to text."""
import scipy.io.wavfile as wav
from deepspeech.model import Model


def build_model(init_settings):
    """

    :param init_settings: Configparser
    :return:
    """
    print('Loading DeepSpeech Models')
    try:
        ds = Model(init_settings['deepspeech']['model_path'],
                   init_settings['deepspeech']['N_FEATURES'],
                   init_settings['deepspeech']['N_CONTEXT'],
                   init_settings['deepspeech']['alphabet_path'],
                   init_settings['deepspeech']['BEAM_WIDTH'])
        ds.enableDecoderWithLM(init_settings['deepspeech']['alphabet_path'],
                               init_settings['deepspeech']['lm_path'],
                               init_settings['deepspeech']['trie_path'],
                               init_settings['deepspeech']['LM_WEIGHT'],
                               init_settings['deepspeech']['WORD_COUNT_WEIGHT'],
                               init_settings['deepspeech']['VALID_WORD_COUNT_WEIGHT'])
        return ds
    except Exception as e:
        print('Loading Error!')
        print(e.message)
        return None

def load_audio(audio_file):
    fs, audio = wav.read(audio_file)
    assert fs == 16000, "Only 16000Hz input WAV files are supported for now!"
    return audio, fs

def ds_infer(ds, audio, fs):
    """
    inference the audio to text by DeepSpeech models
    :param ds: deep speech model
    :param audio:
    :return:
    """
    if not ds:
        return None
    else:
        return ds.stt(audio, fs)