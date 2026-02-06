# This script prepares the tensor data
# for the ASTForAudioClassification model
# from HuggingFace. The particular model
# requires a very particular optimization
# process for the tensors. As a result,
# we use the corresponding ASTFeatureExtractor
# tool, as used in the tutorial in the link
# below:
# https://github.com/NielsRogge/Transformers-Tutorials/tree/master/AST
#
# Important notice!!
# -------------------
# The number of segments in this script should be 3.
# This is because ASTFeatureExtractor exports a fixed
# value of time_frames that should be 1024. This
# is what the ASTForAudioClassification is trained to
# expect as input. Since the sample rate is also fixed
# at 1600 and the hop_size is also fixed at 160, it
# can be calculated that the audio sample duration
# should be approximately 10 seconds. The GTZAN samples
# are of 30 seconds. As a result, the segments should
# be 3.

import os
import torch
import torchaudio
from torchaudio.transforms import Resample
import json
from transformers import ASTFeatureExtractor


# Set DATASET_PATH according to the name of the dataset folder
# Set name of extracted jason file in JSON_PATH
DATASET_PATH = "../.genres_full"
EXPORT_PATH = "backup data/data_full_AST_hf_10_seg.pt"

SAMPLE_RATE = 22050
DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = 16000 * DURATION

# Creating resampling transform
AST_SAMPLE_RATE = 16000
resample = Resample(SAMPLE_RATE, AST_SAMPLE_RATE)

def save_melspec(dataset_path,
                export_path,
                num_segments=5):  #num_segments -> number of pieces that each sound will be chopped into. Because we only have 100 samples for each genre

    # dictionary to store data
    X = []
    y = []
    mapping = []

    # Creating a feature extractor
    feature_extractor = ASTFeatureExtractor()

    # defining variables that will be used later on
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_window_frames_per_segment = 1024 # This is the number of window_frames the ASTForAudioClassification expects

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate (os.walk(dataset_path)):

        # ensure that we're not at the root level
        if dirpath != dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("\\") # genre/blues => ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            mapping.append(semantic_label)
            print(f"\n Processing {semantic_label}")

            # process files for a specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, _ = torchaudio.load(file_path)
                signal = resample(signal)

                # process segments extracting mel spectogram and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # example: s=0 -> 0
                    finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment

                    segment = signal[:, start_sample:finish_sample]

                    # Following the resampling procedure from the documentation tutorial
                    # segment = resample(segment)
                    segment = segment.squeeze().numpy()

                    inputs = feature_extractor(segment,
                                               sampling_rate=AST_SAMPLE_RATE,
                                               padding="max length",
                                               return_tensors="pt")
                    input_values = inputs.input_values.squeeze()

                    # store mel spectogram for segment if it has the expected length. Our inputs need fixed length
                    if len(input_values) == expected_window_frames_per_segment:
                        X.append(input_values)  # ΟΧΙ .tolist()
                        y.append(i - 1)
                        print(f"{file_path}, segment: {s}")

    X = torch.stack(X)  # (N, 1024)
    y = torch.tensor(y, dtype=torch.long)

    torch.save(
        {
            "X": X,
            "y": y,
            "mapping": mapping
        },
        export_path
    )

if __name__ == "__main__":
    save_melspec(DATASET_PATH, EXPORT_PATH, num_segments=10)