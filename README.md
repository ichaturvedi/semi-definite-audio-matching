Predicting emotional state from speech
===
This code implements the model discussed in the paper Sound Identification using Affective. It uses the Affectivespace of emotions to create a prior model for Speech classification into emotions. An audio matching metric is proposed to select the best augmentations for the task. 

Requirements
---
This code is based on the pix2pix code found at:
https://www.mathworks.com/help/deeplearning/ug/sequential-feature-selection-for-speech-emotion-recognition.html

Spanish video reviews
---
![affectivespace](https://user-images.githubusercontent.com/65399216/141210623-89cd06ad-bb20-4c24-9d5d-768d6d9136ed.jpeg)

Preprocessing
---
- The training audios are in the form of wav files (see sample_audio folder).
- The training labels must be in the form of 'Speaker ID, Emotion' (see sample_labels.txt)

Augmentations
---
create_augmentations(datasetFolder, labelsfile, samplingfreq, outputFolder, outputlabels, numAugmentations)
- Training audios will be in the folder datasetFolder
- Labels for training audios will be in labelsfile
- The sampling frequency samplingfreq of an audio can be determined using [aud, samplingfreq] = audioread('sample.wav')
- The number of augmentations is a positive integer numAugmentations
- The augmented audio are stored in outputFolder
- The labels for augmented dataset will be written to outputlables

Audio Matching
---
match = audio_matching(datasetFolder,samplingfreq, goldAudio)
- Augmented audios will be in datasetFolder
- The sampling frequency is an integer samplingfreq
- A clear audio from original dataset is given as goldAudio
- The matching error between goldAudio and datasetFolder is returned as a vector match


Training
---
fmea = speech_classifier(datasetFolder, labelsfile, samplingfreq, priornet, outputnet)
- Training audios will be in datasetFolder
- Labels for training audios will be in labelsfile
- The sampling frequency is an integer samplingfreq 
- Prior speech classifier trained on Affectivespace is give as priornet
- Model trained will be stored in outputnet
- F-measure of each class is written to fmeasure.txt and returned by the function
