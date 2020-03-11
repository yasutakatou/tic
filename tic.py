# -*- coding: utf-8 -*
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import scipy.io.wavfile
import simpleaudio
import soundfile as sf
import sys
import wave 
from art import *
import os
import random

def main():
	angerCount = 0
	args = sys.argv
	picturePath = './pic/'

	try:
		f = sf.SoundFile(args[1])
	except Exception as e:
		viewUsecase()
	
	if len(sys.argv) < 3:
		viewUsecase()

	if int(args[2]) < 1 or int(args[2]) > 100:
		viewUsecase()

	waveToPng(args[1],"target.png")
	seconds = len(f) / f.samplerate
	print('seconds = ', seconds)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	detector = cv2.ORB_create()
	(target_kp, target_des) = calc_kp_and_des("target.png", detector)

	try:
		while True:
			recordSound("now.wav", seconds / 2)
			waveToPng("now.wav","now.png")
			try:
				(comparing_kp, comparing_des) = calc_kp_and_des("now.png", detector)
				matches = bf.match(target_des, comparing_des)
				dist = [m.distance for m in matches]
				ret = sum(dist) / len(dist)
			except cv2.error:
				ret = 100000

			print(ret, ' match!')
			if float(args[2]) > ret:
				playInvertSound(args[1])
				angerCount = angerCount + 1
				tprint(str(angerCount) + ' anger!',"random")
			else:
				simpleaudio.stop_all()

	except KeyboardInterrupt:
		print('Exit.')

def viewUsecase():
	print("usecase: python tic.py arg1:(wave file) arg2:(100 > detect value > 1)")
	sys.exit()

def playInvertSound(waveFile):
	soundTmp = AudioSegment.from_mp3(waveFile)
	sound = soundTmp.invert_phase()
	playback = simpleaudio.play_buffer(
		sound.raw_data, 
		num_channels=sound.channels, 
		bytes_per_sample=sound.sample_width, 
		sample_rate=sound.frame_rate
	)

def calc_kp_and_des(img_path, detector):
	IMG_SIZE = (200, 200)
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, IMG_SIZE)
	return detector.detectAndCompute(img, None)

def waveToPng(importWave, exportImage):
	rate, data = scipy.io.wavfile.read(importWave)
	data = data / 32768
	fft_size = 1024                 
	hop_length = int(fft_size / 4)  
	amplitude = np.abs(librosa.core.stft(data, n_fft=fft_size, hop_length=hop_length))
	log_power = librosa.core.amplitude_to_db(amplitude)
	librosa.display.specshow(log_power, sr=rate, hop_length=hop_length, cmap='magma')
	plt.savefig(exportImage)

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
