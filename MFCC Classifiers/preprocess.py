import os
import torch
import torchaudio
import math
import json

# Set DATASET_PATH according to the name of the dataset folder
# Set name of extracted jason file in JSON_PATH
DATASET_PATH = "../genres_small"
JSON_PATH = "data_small.json"

SAMPLE_RATE = 22050
DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path,
              json_path,
              n_mfcc=13,
              n_fft=2048,
              hop_length=512,
              num_segments=5):  #num_segments -> number of pieces that each sound will be chopped into. Because we only have 100 samples for each genre

    # dictionary to store data
    data = {
        "mapping" : [],
        "mfcc" : [],
        "labels" : [],
    }

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

                # create an mfcc transform for the following for loop
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=sr,
                    n_mfcc=n_mfcc,
                    melkwargs={
                        "n_fft": n_fft,
                        "hop_length": hop_length,
                        "n_mels": 128  # this is the librosa default. could be 40
                    }
                )

                # process segments extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # example: s=0 -> 0
                    finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment

                    segment = signal[:, start_sample:finish_sample]
                    mfcc = mfcc_transform(segment)
                    mfcc = mfcc.squeeze(0).T    # Using .T is crucial for input shape for RNNs. Optional for CNNs.

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_window_frames_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1) # the first iteration is for the dataset path so we ignore it with (i-1)
                        print(f"{file_path}, segment: {s}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)