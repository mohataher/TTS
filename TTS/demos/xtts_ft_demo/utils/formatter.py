import os
import gc
import torchaudio
import pandas
from faster_whisper import WhisperModel
from glob import glob

from tqdm import tqdm

import torch
import torchaudio
# torch.set_num_threads(1)


from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

torch.set_num_threads(16)


import os


audio_types = (".wav", ".mp3", ".flac")


from seamless_communication.inference import Translator
AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "Egyptian Arabic"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32
  
dtype = torch.float16




def preprocess_audio(input_audio: str) -> None:
    arr, org_sr = torchaudio.load(input_audio)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        print(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    torchaudio.save(input_audio, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))



def run_asr(translator, input_audio: str, target_language: str) -> str:
    preprocess_audio(input_audio)
    target_language_code = target_language # LANGUAGE_NAME_TO_CODE[target_language]
    out_texts, _ = translator.predict(
        input=input_audio,
        task_str="ASR",
        src_lang=target_language_code,
        tgt_lang=target_language_code,
    )
    return str(out_texts[0])




def list_audios(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=audio_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an audio and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the audio and yield it
                audioPath = os.path.join(rootDir, filename)
                yield audioPath

def format_audio_list(audio_files, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    audio_total_size = 0
    # make sure that ooutput file exists
    os.makedirs(out_path, exist_ok=True)

    # Loading Whisper
    #device = "cuda" if torch.cuda.is_available() else "cpu" 

    print("Loading running ASR seamless")
    translator = Translator(
      model_name_or_card="seamlessM4T_v2_large",
      vocoder_name_or_card="vocoder_v2",
      device=device,
      dtype=dtype,
      apply_mintox=True,
    )
    #asr_model = WhisperModel("asr-whisper-large-v2-commonvoice-ar", device=device, compute_type="float16")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    i=0
    for audio_path in tqdm_object:
        wav, sr = torchaudio.load(audio_path)
        # stereo to mono if needed
        if wav.size(0) != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        wav = wav.squeeze()
        audio_total_size += (wav.size(-1) / sr)

        #segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
        segments = run_asr(translator, audio_path, target_language='arz')
        print(segments)
        

        sentence = segments
        # Expand number and abbreviations plus normalization
        sentence = multilingual_cleaners(sentence, target_language)
        audio_file_name, _ = os.path.splitext(os.path.basename(audio_path))
        i += 1
        audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"
        
        absoulte_path = os.path.join(out_path, audio_file)
        os.makedirs(os.path.dirname(absoulte_path), exist_ok=True)
        first_word = True

        audio = wav.unsqueeze(0)
        # if the audio is too short ignore it (i.e < 0.33 seconds)
        #if audio.size(-1) >= sr/3:
        torchaudio.save(absoulte_path,
            audio,
            sr
        )
        #else:
        #    print('skipped audio')
        #    continue

        metadata["audio_file"].append(audio_file)
        metadata["text"].append(sentence)
        metadata["speaker_name"].append(speaker_name)

    df = pandas.DataFrame(metadata)
    df = df.sample(frac=1)
    num_val_samples = int(len(df)*eval_percentage)

    df_eval = df[:num_val_samples]
    df_train = df[num_val_samples:]

    df_train = df_train.sort_values('audio_file')
    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)

    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_eval = df_eval.sort_values('audio_file')
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    # deallocate VRAM and RAM
    del translator, df_train, df_eval, df, metadata
    gc.collect()

    return train_metadata_path, eval_metadata_path, audio_total_size
