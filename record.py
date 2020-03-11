# -*- coding: utf-8 -*
import pyaudio
import wave 
import numpy as np
import os
import sys
from PIL import Image

def main():
	args = sys.argv
	
	if len(sys.argv) < 3:
		viewUsecase()

	if int(args[2]) < 4 or int(args[2]) > 100:
		viewUsecase()

	recordSound(args[1], int(args[2]))

def viewUsecase():
	print("usecase: python tic.py arg1:(wave file) arg2:(100 > record seconds > 4)")
	sys.exit()

def recordSound(img_path, seconds):
	fmt = pyaudio.paInt16
	ch = 1
	sampling_rate = 44100
	chunk = 2**11
	audio = pyaudio.PyAudio()
	index = 1
	stream = audio.open(format=fmt, channels=ch, rate=sampling_rate, input=True,
	                    input_device_index = index,
	                    frames_per_buffer=chunk)

	print("recording start...")

	frames = []

	for i in range(0, int(sampling_rate / chunk * seconds)):
	    buf = stream.read(chunk)
	    frames.append(buf)
	    data = np.fft.fft(np.frombuffer(buf, dtype="int16"))

	print("recording  end...")

	stream.stop_stream()
	stream.close()
	audio.terminate()

	wav = wave.open(img_path, 'wb')
	wav.setnchannels(ch)
	wav.setsampwidth(audio.get_sample_size(fmt))
	wav.setframerate(sampling_rate)
	wav.writeframes(b''.join(frames))
	wav.close()

if __name__ == '__main__':
    main()
