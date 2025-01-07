# speaker-identification-abi-1

A Speech Technology project at University of Malta (2023-2024).

This project explores the application of various model architectures to the task of Speaker Identification (SID). The study was carried out on the audio data from the <em>Accents of the British Isles (ABI-1) Corpus</em>, which includes 284 speakers belonging to 14 different accents.

First, Mel-spectrograms are extracted for each 3-second chunk of every audio with the help of <em>librosa</em> library. Then, the data is converted into tensors and prepared for training. A confusion matrix is constructed for the results of each model.

Three model architectures are investigated in this project: a shallow architecture, a Very Deep Convolutional Network (VGG) and VGG + LSTM architecture. The models are implemented using  Keras.

## 1. A shallow model.

<p align="center">

| No.   | Layer                 | Details              |
|-------|-----------------------|----------------------|
| 1     | `Conv2D`              | (3, 3), 64 filters   |
| 2     | `ReLU`                |                      |
| 3     | `BatchNormalization2d`|                      |
| 4     | `MaxPool2d`           | (3, 3)               |
| 5     | `Conv2D`              | (3, 3), 128 filters  |
| 6     | `ReLU`                |                      |
| 7     | `BatchNormalization2d`|                      |
| 8     | `MaxPool2d`           | (3, 3)               |
| 9     | `Linear`              | 2048 neurons         |
| 10    | `ReLU`                |                      |
| 11    | `Dropout`             | 0.5                  |
| 12    | `Linear`              | 284 neurons          |

</p>

The shallow model achieved:
*  the accuracy of 0.54 on the test set;
*  the F1-score of 0.51 on the test set.


## 2. VGG (Very Deep Convolutional Network).

| No.   | Layer                 | Details              |
|-------|-----------------------|----------------------|
| 1     | `Conv2D`              | (3, 3), 32 filters   |
| 2     | `ReLU`                |                      |
| 3     | `BatchNormalization2d`|                      |
| 4     | `Conv2D`              | (3, 3), 32 filters   |
| 5     | `ReLU`                |                      |
| 6     | `BatchNormalization2d`|                      |
| 7     | `MaxPool2d`           | (2, 2)               |
| 8     | `Conv2D`              | (3, 3), 64 filters   |
| 9     | `ReLU`                |                      |
| 10    | `BatchNormalization2d`|                      |
| 11    | `Conv2D`              | (3, 3), 64 filters   |
| 12    | `ReLU`                |                      |
| 13    | `BatchNormalization2d`|                      |
| 14    | `MaxPool2d`           | (2, 2)               |
| 15    | `Conv2D`              | (3, 3), 128 filters  |
| 16    | `ReLU`                |                      |
| 17    | `BatchNormalization2d`|                      |
| 18    | `Conv2D`              | (3, 3), 128 filters  |
| 19    | `ReLU`                |                      |
| 20    | `BatchNormalization2d`|                      |
| 21    | `Conv2D`              | (3, 3), 128 filters  |
| 22    | `ReLU`                |                      |
| 23    | `BatchNormalization2d`|                      |
| 24    | `MaxPool2d`           | (2, 2)               |
| 25    | `Linear`              | 1024 neurons         |
| 26    | `ReLU`                |                      |
| 27    | `Dropout`             | 0.5                  |
| 28    | `Linear`              | 1024 neurons         |
| 29    | `ReLU`                |                      |
| 30    | `Dropout`             | 0.5                  |
| 31    | `Linear`              | 284 neurons          |

The VGG model achieved:
*  the accuracy of 0.82 on the test set;
*  the F1-score of 0.81 on the test set.


## 3. VGG + LSTM.

| No.   | Layer                 | Details              |
|-------|-----------------------|----------------------|
| 1     | `Conv2D`              | (3, 3), 32 filters   |
| 2     | `ReLU`                |                      |
| 3     | `BatchNormalization2d`|                      |
| 4     | `Conv2D`              | (3, 3), 32 filters   |
| 5     | `ReLU`                |                      |
| 6     | `BatchNormalization2d`|                      |
| 7     | `MaxPool2d`           | (2, 2)               |
| 8     | `Conv2D`              | (3, 3), 64 filters   |
| 9     | `ReLU`                |                      |
| 10    | `BatchNormalization2d`|                      |
| 11    | `Conv2D`              | (3, 3), 64 filters   |
| 12    | `ReLU`                |                      |
| 13    | `BatchNormalization2d`|                      |
| 14    | `MaxPool2d`           | (2, 2)               |
| 15    | `Conv2D`              | (3, 3), 128 filters  |
| 16    | `ReLU`                |                      |
| 17    | `BatchNormalization2d`|                      |
| 18    | `Conv2D`              | (3, 3), 128 filters  |
| 19    | `ReLU`                |                      |
| 20    | `BatchNormalization2d`|                      |
| 21    | `Conv2D`              | (3, 3), 128 filters  |
| 22    | `ReLU`                |                      |
| 23    | `BatchNormalization2d`|                      |
| 24    | `MaxPool2d`           | (2, 2)               |
| 25    | `Flatten`             |                      |
| 26    | `LSTM`                | 512 hidden size      |
| 27    | `Linear`              | 512 neurons          |
| 28    | `ReLU`                |                      |
| 29    | `Dropout`             | 0.5                  |
| 30    | `Linear`              | 512 neurons          |
| 31    | `ReLU`                |                      |
| 32    | `Dropout`             | 0.5                  |
| 33    | `Linear`              | 284 neurons          |

The VGG + LSTM model achieved:
*  the accuracy of 0.72 on the test set;
*  the F1-score of 0.71 on the test set.

## Model comparison

| Model      | Accuracy (test set)   | F1-score (test set)                 
|------------|-----------------------|----------------------|
| Shallow    | 0.54                  | 0.51                 |
| VGG        | 0.82                  | 0.81                 |
| VGG + LSTM | 0.72                  | 0.71                 |