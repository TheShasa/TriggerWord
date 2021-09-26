import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment
import utils.config as config
# Calculate and plot spectrogram for a wav audio file


def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = config.NFTT  # Length of each window segment
    fs = config.FS  # Sampling frequencies
    noverlap = config.N_OVERLAP  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(
            data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx

# Load a wav file


def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis


def load_raw_audio(backgroud_folder, activate_folder,
                   negative_folder):
    activates = get_wavs_from_folder(activate_folder)
    backgrounds = get_wavs_from_folder(backgroud_folder)
    negatives = get_wavs_from_folder(negative_folder)
    return activates, negatives, backgrounds


def get_wavs_from_folder(folder):
    res = []
    for filename in os.listdir(folder):
        if filename.endswith("wav"):
            audio = AudioSegment.from_wav(folder+filename)
            res.append(audio)
    return res
