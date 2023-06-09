from pyaudio import PyAudio, paInt16
import io
import numpy as np
import torch
import torchaudio
torchaudio.set_audio_backend('soundfile')
import scipy.io.wavfile as wav
from collections import deque
import threading
import time


class Listener:
    def __init__(self, chunk_size=1024, sample_rate=16000, duration=2):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.duration = duration

        self.stream = PyAudio().open(format=paInt16, channels=1, rate=16000, input=True, frames_per_buffer=self.chunk_size)
        self.queue = deque(maxlen=duration * sample_rate // chunk_size)
        self._listenting = True
    
    @property
    def listenting(self):
        return self._listenting

    def listen(self):
        while self.listenting:
            chunk = self.stream.read(self.chunk_size)
            self.queue.append(chunk)
    
    def start(self):
        self._listenting = True
    
    def stop(self):
        self._listenting = False

    def clear(self):
        self._listenting = False
        self.queue.clear()
        self._listenting = True

    def run(self):
        thread = threading.Thread(target=self.listen, daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening... \n")
        return self.queue
        

class Runner:
    def __init__(self):
        self.model = torch.jit.load('wake_words/compiled/epoch_0.pt')
        pass

    def run(self):
        thread = threading.Thread(target=self._loop)
        thread.start()

    def _loop(self):
        sample_rate = 16000
        listener = Listener(2048, sample_rate, duration=3)
        stream = listener.run()
        i = 0
        while True:
            if len(stream) < stream.maxlen:
                continue
            bytes_io = io.BytesIO()
            print(np.frombuffer(stream[0]).shape)
            raw_data = np.frombuffer(b''.join(stream), dtype=np.int16)
            wav.write(bytes_io, sample_rate, raw_data)
            waveform, _ = torchaudio.load(bytes_io)

            pred, prob = self.model([waveform])
            if pred.item() == 1 and prob.item() > 0.9:
                print(f'Wake Word Detected with Probability {prob.item()}')
                torchaudio.save(f'wake_words/dump/{i}.wav', waveform, sample_rate)
                i += 1
                listener.clear()
            time.sleep(0.1)

Runner().run()