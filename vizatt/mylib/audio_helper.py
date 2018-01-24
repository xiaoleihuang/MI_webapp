"""This script is to use the locally installed Mozilla Deepseech package to convert input audio to text."""
import scipy.io.wavfile as wav
from deepspeech.model import Model
import gtts

def build_model(init_settings):
    """

    :param init_settings: Configparser
    :return:
    """
    print('Loading DeepSpeech Models')
    try:
        ds = Model(str(init_settings['deepspeech']['model_path']),
                   int(init_settings['deepspeech']['N_FEATURES']),
                   int(init_settings['deepspeech']['N_CONTEXT']),
                   str(init_settings['deepspeech']['alphabet_path']),
                   int(init_settings['deepspeech']['BEAM_WIDTH']))
        ds.enableDecoderWithLM(str(init_settings['deepspeech']['alphabet_path']),
                               str(init_settings['deepspeech']['lm_path']),
                               str(init_settings['deepspeech']['trie_path']),
                               float(init_settings['deepspeech']['LM_WEIGHT']),
                               float(init_settings['deepspeech']['WORD_COUNT_WEIGHT']),
                               float(init_settings['deepspeech']['VALID_WORD_COUNT_WEIGHT']))
        return ds
    except Exception as e:
        print('Loading Error!')
        print(e)
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

def text2audio(input_str, file_name='tts.mp3'):
    tts = gtts.gTTS(text=input_str, lang='en', slow=False)
    tts.save('./resources/audio_tmp/'+file_name)