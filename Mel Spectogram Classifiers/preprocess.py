import os
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB
import math
import json

# Set DATASET_PATH according to the name of the dataset folder
# Set name of extracted jason file in JSON_PATH
DATASET_PATH = "../genres_full"
JSON_PATH = "data_full.json"

SAMPLE_RATE = 22050
DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_melspec(dataset_path,
                json_path,
                n_mels=64,
                n_fft=4096,
                win_length=1024, # This is proposed in this paper: https://www.mdpi.com/2227-7390/10/23/4427 (It usually is equal to n_fft)
                hop_length=512,
                num_segments=5):  #num_segments -> number of pieces that each sound will be chopped into. Because we only have 100 samples for each genre

    # dictionary to store data
    data = {
        "mapping" : [],
        "melspec" : [],
        "labels" : [],
    }

    # defining variables that will be used later on
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_window_frames_per_segment = math.ceil(num_samples_per_segment / hop_length) # round this number 1.2 -> 2

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate (os.walk(dataset_path)):

        # ensure that we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("\\") # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print(f"\n Processing {semantic_label}")

            # process files for a specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = torchaudio.load(file_path) # RESAMPLING IS NOT NEEDED. GTZAN USES sr = 22050 Hz

                '''
                Important note:
                
                The typical steps for creating a mel spectogram are:
                1. Extract Spectogram using Short-Time Fourier Transform
                2. Convert amplitude to DBs
                3. Convert fequencies to Mel Scale:
                    i. Choose numer of mel bands
                    ii. Construct mel filter banks
                    iii. Apply mel filter banks to spectogram
                    
                In the code we appear to follow the steps 1 -> 3 -> 2.
                This is because we follow the PyTorch tutorial guide on how to extract audio features
                and in particular, Mel Spectograms. The guide mentions extracting the mel spectogram tensor
                and then passing it through the amplitude_to_DB() method included in the plot_spectrogram()
                function that is used to project spectograms.
                 
                The guide is in the following link:
                https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
                
                '''

                # create a mel spectogram transform for the following for loop
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    n_mels=n_mels
                )

                # create a transform to turn the mel spectogram tensor from the power/amplitude scale to the decibel scale.
                power_to_db = AmplitudeToDB("power", 80.0)

                # process segments extracting mel spectogram and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # example: s=0 -> 0
                    finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment

                    segment = signal[:, start_sample:finish_sample]
                    melspec  = mel_transform(segment)
                    # Similar way to do the db transform, produces almost identical results:
                    # melspec = torchaudio.functional.amplitude_to_DB(melspec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
                    melspec = power_to_db(melspec)
                    melspec = melspec.squeeze(0).T    # Using .T is crucial for input shape for RNNs. Optional for CNNs.

                    # store mel spectogram for segment if it has the expected length. Our inputs need fixed length
                    if len(melspec) == expected_window_frames_per_segment:
                        data["melspec"].append(melspec.tolist())
                        data["labels"].append(i-1) # the first iteration is for the dataset path so we ignore it with (i-1)
                        print(f"{file_path}, segment: {s}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_melspec(DATASET_PATH, JSON_PATH, num_segments=10)