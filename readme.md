# NLP for ECOSud Project - "Jules Verne"

Self explanatory 'data' and 'scripts' folders.


## Library requirements

To review and edit the alignments, and to extract pitch and formant data we used Praat (6.1.38)

Here's an incomplete list of all the python libraries:

- MFA - Montreal Force Alignment (2.0). (Separate python environment encouraged). 

`conda config --add channels conda-forge`

`conda install montreal-forced-aligner`

After installation, download the english model and dictionary: 

`mfa models download acoustic english_mfa`

`mfa models download dictionary english_mfa`

- Torch (1.11.0) - `pip install torch`

- Transformers (4.20.0) - from HuggingFace. `pip install transformers`

- Whisper () - `pip install openai-whisper`



## Analysis pipeline

- Transcribe audio .wav files with Whisper.

Example:

`whisper trial01.wav --language en --model medium --output_dir ./ --output_format json `

this might take a while... try "--model small" instead and check results

all files at once:

`for i in {01..16}; do whisper trial$i.wav --language en --model medium --output_dir ./ --output_format json; done` 


- Run process_json.py to extract json files to plain text

- Manually inspect and modify transcription plain text files if necessary (it probably is). Make sure these files have ONLY ONE LINE of text!!

- Move revised plain text files to the "wavfiles" folder in 'data'. Rename them to match the name of each wavefile (keep the .txt extension). The structure should be something like `./data/wavfiles/trial{01..11}.{txt,wav}`

- (Separate python environment encouraged) Run mfa to validate and align transcriptions to wav files. 

Example: 

`mfa validate ./wav english_mfa english_mfa`

`mfa align --clean ./wav english_mfa english_mfa ./wav/aligned `


- Open each wav and textGrid file with Praat and revise for inconsistencies and misalignments (if you want a complete annotation of all the phonemes, that's going to take some work...)

- Save the revised `textGrid` files to the `./revised` subfolder. Change the file encoding from UFT-16 to UtF-8 (using Kate, Notepad++ or your favourite plain text editor)

- Run `embed_main.py`

- There should be now a file named `embed.mat` in the data folder

    The structure is as follows:
    
    - A n-by-1 cell array with each filename
    
    - A n-by-1 cell array with morphoeme markings. Each cell contains a timepoints-by-2 matrix. onset and offset are the columns
    
    - A cell array with morphemes. Each cell contains a char-array of timepoints-by-width, where width is the max number of characters (uses right-padding). Ideally should be one... but maybe not
    
    - A cell array with prediction "probabilities". Each cell contains a timepoints-by-3 matrix: onset, offset and "prob" and the columns.
    
    - A n-by-1 cell array with the text markings. Each cell contains a char-array of timepoints-by-width, where width is the max number of characters (uses right-padding).
        
To obtain pitch and forman data from the audio files:

- Open all audio files in Praat (all at once if you're feeling lucky)

- Open the pitch_extractor.praat script

- Run the script once for each audio file (you should write the name of the corresponding Sound object as it appears on the list, but without the number. For instance: Sound renard1)

- You can provide a corresponding output filename (renard_pitch.csv)

- I'm not sure where Praat is going to create each output file. Probably in either a) the Home folder, or b) the last folder in which you saved a file during this session. You can save a bogus file at the corresponding folder just to make things easy when you export the csv files.

- After processig each Sound object, remove it, as well as its newly created Formant and Pitch objects (may not be necessary but it's cleaner, right?)

- Have fun!
