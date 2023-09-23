from  transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

model_name = "facebook/musicgen-small"

processor = AutoProcessor.from_pretrained(model_name)
model = MusicgenForConditionalGeneration.from_pretrained(model_name)

inputs = processor(
    text=["A slow scottish aire played on a fiddle"],
    padding = True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=(256))
sampling_rate = model.config.audio_encoder.sampling_rate

scipy.io.wavfile.write("musicgen_out.wav", rate = sampling_rate, data = audio_values[0, 0].numpy())
