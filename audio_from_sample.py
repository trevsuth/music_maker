from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
import scipy.io.wavfile as wav

model="facebook/musicgen-small"

processor = AutoProcessor.from_pretrained(model)
model = MusicgenForConditionalGeneration.from_pretrained(model)

# Read the .wav file
filename = "musicgen_output.wav"  # replace with the path to your .wav file
sampling_rate, sample = wav.read(filename)

# If the audio file has multiple channels (e.g., stereo), select one of the channels (e.g., the first one)
#if len(sample.shape) > 1:
#    sample = sample[:, 0]

# Create segments from the audio sample, e.g., first quarter and first half
#sample_1 = sample[: len(sample) // 4]
#sample_2 = sample[: len(sample) // 2]

# Prepare the input data
inputs = processor(
    audio=sample,
    sampling_rate=sampling_rate,
    text='harmonic cello backup',
    padding=True,
    return_tensors="pt",
)

# Generate new audio samples
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
sampling_rate = model.config.audio_encoder.sampling_rate

wav.write("musicgen_out_sample.wav", rate = sampling_rate, data = audio_values[0, 0].numpy())
