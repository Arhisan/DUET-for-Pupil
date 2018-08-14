import wave
import numpy as np


class audioprocessor:
    def __init__(self, filename):
        self.filename = filename

    types = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }

    def get_spectrum(self):
        wav = wave.open(self.filename, mode="r")
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
        content = wav.readframes(nframes)
        samples = np.fromstring(content, dtype = self.types[sampwidth])
        return samples


