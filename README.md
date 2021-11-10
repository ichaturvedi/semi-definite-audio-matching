Predicting emotional state from speech
===
This code implements the model discussed in the paper Sound Identification using Affective. It uses the Affectivespace of emotions to create a prior model for Speech classification into emotions. An audio matching metric is proposed to select the best augmentations for the task. 

Requirements
---
This code is based on the pix2pix code found at:
https://www.mathworks.com/help/deeplearning/ug/sequential-feature-selection-for-speech-emotion-recognition.html

Preprocessing
---

Augmentations
---
matlab -r create_augmentations(datasetFolder, labelsfile, samplingfreq, outputFolder, outputlabels, numAugmentations)
- Training audios will be in the folder datasetFolder
- Labels for training audios will be in labelsfile
- The sampling frequency samplingfreq of an audio can be determined using [aud, samplingfreq] = audioread('sample.wav')
- The number of augmentations is a positive integer numAugmentations
- The augmented audio are stored in outputFolder
- The labels for augmented dataset will be written to outputlables

Audio Matching
---

Training
---
