# music_maker

## Description
Using facebook/musicgen-small to make small music samples

The core file is musicgen_cli.py, though there is a rough streamlit app as well

## Use
1. From the command line, run `python musicgen_cli.py "a slow Scottish fiddle tune"`
2. This will generate 2 wav files, one that is the initial creation (musicgen_output.wav) and a second, longer one that is derived from the first (sample_output.wav).
3. Alter the model names in musicgen_cli.py to use larger models (e.g. change "facebook/musicgen-small" to "facebook/musicgen-large")
4. To use the streamlit app, run `streamlit run app.py`

## To Do
- Update requirements.txt and create install script
- Break out model size (Small, Med, Large, Melody) to function call for easier use
- Improve streamlit interface
- Experiment with model params (temperature, etc)

## References
- https://huggingface.co/facebook/musicgen-small
- https://huggingface.co/docs/transformers/model_doc/musicgen
- https://github.com/facebookresearch/audiocraft
- https://arxiv.org/pdf/2306.05284.pdf
