import streamlit as st
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wav
import tempfile
import io
import numpy as np

def make_audio_from_text(model_prompt, output_file='', return_token_lengh=265):
    """Makes an audio file from a text prompt.
    
    Keyword arguments:
    model_prompt -- text input used to generate the audio file
    output_file -- if not set or set with a value of '' then
        will return a tuple of (audio_data, sampling_data).  If set 
        to anything else will save the audio file in wav format to
        that path
    return_token_length -- defaults to 256 (~5 seconds of music).  Larger
        values will create longer music outputs, though longer files then to be
        less aurally interesting
    """
    #declare model name
    model_name = "facebook/musicgen-small"
    
    #make calls to hugging face
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    #model specific inputs
    inputs = processor(
        text=[model_prompt],
        padding=True,
        return_tensors="pt",
    )

    #generate audio sample
    audio_values = model.generate(**inputs, max_new_tokens=return_token_lengh)
    sampling_rate = model.config.audio_encoder.sampling_rate

    #If no outputfile is specified, return the audio data, otherwise save the file
    if output_file=='':
        return (audio_values, sampling_rate)
    else:
        #save return to file
        save_wav_file(music_data=audio_values, 
                      sampling_rate=sampling_rate,
                      file_name=output_file)

def make_audio_from_sample(model_prompt, audio_sample, output_file='', return_token_lengh=265):
    """Makes an audio file from a text prompt.
    
    Keyword arguments:
    model_prompt -- text input used to generate the audio file
    audio_sample -- a wav file so use as a starting sample
    output_file -- if not set or set with a value of '' then
        will return a tuple of (audio_data, sampling_data).  If set 
        to anything else will save the audio file in wav format to
        that path
    return_token_length -- defaults to 256 (~5 seconds of music).  Larger
        values will create longer music outputs, though longer files then to be
        less aurally interesting
    """
    
    model="facebook/musicgen-small"

    processor = AutoProcessor.from_pretrained(model)
    model = MusicgenForConditionalGeneration.from_pretrained(model)

    # Read the .wav file
    sampling_rate, sample = read_wav_file(audio_sample)  # replace with the path to your .wav file

    # Create segments from the audio sample, e.g., first quarter and first half
    #sample_1 = sample[: len(sample) // 4]
    #sample_2 = sample[: len(sample) // 2]

    # Prepare the input data
    inputs = processor(
        audio=sample,
        sampling_rate=sampling_rate,
        text=model_prompt,
        padding=True,
        return_tensors="pt",
    )

    # Generate new audio samples
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=return_token_lengh)
    sampling_rate = model.config.audio_encoder.sampling_rate

    #If no outputfile is specified, return the audio data, otherwise save the file
    if output_file=='':
        return (audio_values, sampling_rate)
    else:
        #save return to file
        save_wav_file(music_data=audio_values, 
                      sampling_rate=sampling_rate,
                      file_name=output_file)

def save_wav_file(music_data, sampling_rate, file_name='musicgen_out.wav'):
    """Saves music data to wav file.
    
    Keyword Arguments:
    music_data -- the audio data generated from a model
    sampling_rate -- the sampling data generated from a model
    file_name -- the filename that the data will be saved to 
        (defaults to musicgen_out.wav)
    """
    output_file = file_name
    wav.write(output_file, 
              rate=sampling_rate,
              data=music_data[0, 0].numpy())

def read_wav_file(filename):
    """Reads a wav file and returns the sampling rate and audio info"""
    sampling_rate, sample = wav.read(filename)

    # If the audio file has multiple channels (e.g., stereo), select one of the channels (e.g., the first one)
    if len(sample.shape) > 1:
        sample = sample[:, 0]
    
    return (sampling_rate, sample)

def main():
    st.title('Music Generation App')

    # User input
    user_input = st.text_input("Enter the text description for the music:")

    # Button to generate music
    if st.button('Generate Music') and user_input:
        print(user_input)
        # Generate the audio from text
        audio_values, sampling_rate = make_audio_from_text(user_input)
        
        # Convert the audio values to bytes
        audio_bytes = wav_to_bytes(audio_values.numpy(), sampling_rate)
        
        # Play the audio in the app
        #st.audio(audio_bytes, format='audio/wav')
        st.audio(audio_bytes)
        
def wav_to_bytes(audio_array, sample_rate):
    """Converts numpy array to bytes"""
    byte_io = io.BytesIO()
    wav.write(byte_io, sample_rate, np.squeeze(audio_array).astype(np.int16))
    byte_io.seek(0)
    return byte_io.read()

if __name__ == "__main__":
    main()