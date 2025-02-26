#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# Program to Save a 1-second Audio File
import time
import os
import pyaudio
import wave


"""
p = pyaudio.PyAudio()
print("===== PyAudio Device List =====")
device_count = p.get_device_count()
for i in range(device_count):
    info = p.get_device_info_by_index(i)
    print(f"Index: {i}, Name: {info['name']}, Max Input Channels: {info['maxInputChannels']}")
p.terminate()
"""

OUTPUT_WAV_DIR = "wav_outputs"
if not os.path.exists(OUTPUT_WAV_DIR):
    os.makedirs(OUTPUT_WAV_DIR)

DEFAULT_DEVICE_INDEX = 0  # Use the correct device index
channels = 2  # Match the device's capabilities
sample_format = pyaudio.paInt16
fs = 16000
chunk = 1024

def record_one_second_audio():
    wave_filename = "1_second_audio.wav"
    output_path = os.path.join(OUTPUT_WAV_DIR, wave_filename)

    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=sample_format,
            channels=channels,
            rate=fs,
            frames_per_buffer=chunk,
            input=True,
            input_device_index=DEFAULT_DEVICE_INDEX
        )

        print("[Audio] Recording...")
        frames = [stream.read(chunk, exception_on_overflow=False) for _ in range(int(fs / chunk))]
        stream.stop_stream()
        stream.close()

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))

        print(f"[Audio] Saved as {output_path}")

    except Exception as e:
        print(f"[Error] Audio recording failed: {e}")

    finally:
        p.terminate()

if __name__ == "__main__":
    record_one_second_audio()

