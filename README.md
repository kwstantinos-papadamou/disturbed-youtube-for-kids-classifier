# Disturbed YouTube for Kids: Characterizing and Detecting Inappropriate Videos Targeting Young Children

A large number of the most-subscribed YouTube channels target children of very young age. Hundreds of toddler-oriented channels on YouTube feature inoffensive, well produced, and educational videos. Unfortunately, inappropriate content that targets this demographic is also common. YouTube’s algorithmic recommendation system regrettably suggests inappropriate content because some of it mimics or is derived from otherwise appropriate content. Considering the risk for early childhood development, and an increasing trend in toddler’s consumption of YouTube media, this is a worrisome problem.

In this work, we build a classifier able to discern inappropriate content that targets toddlers on YouTube with 84.3% accu- racy, and leverage it to perform a first-of-its-kind, large-scale, quantitative characterization that reveals some of the risks of YouTube media consumption by young children. Our analysis reveals that YouTube is still plagued by such disturbing videos and its currently deployed countermeasures are ineffective in terms of detecting them in a timely manner. Alarmingly, using our classifier we show that young children are not only able, but likely to encounter disturbing videos when they randomly browse the platform starting from benign videos.

This work has been accepted for presentetion at ICWSM 2020. Preprint available here: https://arxiv.org/abs/1901.07046 

## Description
In this repository we include a package with the latest version of the deep learning model implemented in this work which can be used by anyone who wants to detect inappropriate videos for kids on YouTube.

## Model Architecture
![Model Architecture Diagram](https://github.com/kwstantinos-papadamou/disturbed-youtube_videos-detection/blob/master/model_architecture.png)

## Model Description
The classifier consists of four different branches, where each branch processes a distinct feature type: title, tags, thumbnail, and statistics and style features. Then the outputs of all the branches are concatenated to form a two-layer, fully connected neural network that merges their output and drives the final classification.

The title feature is fed to a trainable embedding layer that outputs a 32-dimensional vector for each word in the title text. Then, the output of the embedding layer is fed to a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) that captures the relationships between the words in the title. For the tags, we use an architecturally identical branch trained separately from the title branch.

For thumbnails, due to the limited number of training examples in our dataset, we use transfer learning and the pre-trained Inception-v3 Convolutional Neural Network (CNN), which is built from the large-scale ImageNet dataset.13 We use the pre-trained CNN to extract a meaningful feature representation (2,048-dimensional vector) of each thumbnail. Last, the statistics together with the style features are fed to a fully-connected dense neural network comprising 25 units.

The second part of our classifier is essentially a two-layer, fully-connected dense neural network. At the first layer, (dubbed Fusing Network), we merge together the outputs of the four branches, creating a 2,137-dimensional vector. This vector is subsequently processed by the 512 units of the Fusing Network. Next, to avoid possible over-fitting issues we regularize via the prominent Dropout technique. We apply a Dropout level of d=0.5, which means that during each iterations of training, half of the units in this layer do not update their pa- rameters. Finally, the output of the Fusing Network is fed to the last dense-layer neural network of four units with softmax activation, which are essentially the probabilities that a particular video is suitable, disturbing, restricted, or irrelevant.

## Requirements
1. Python 3.5+
2. Tensorflow 1.13.1
3. Keras 2.2.4
4. Scikit Learn 0.20
5. NLTK 3.4+
6. Numpy 1.14.0
7. Google API Client 1.7.4

## Requirements Installation
1. Install Python 3.5
```
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
wget https://www.python.org/ftp/python/3.5.6/Python-3.5.6.tgz
sudo tar xzf Python-3.5.6.tgz
cd Python-3.5.6
sudo ./configure --enable-optimizations
sudo make altinstall
```

2. Install required libraries
```
pip install --user -U tensorflow==1.13.1
pip install --user -U keras==2.2.4
pip install --user -U scikit-learn
pip install --user -U nltk
pip install --user -U numpy
pip install --user -U isodate emoji requests
pip install --user -U --force-reinstall google-api-python-client
```

OR install using the requirements.txt file:
```
pip install -r requirements.txt --no-index
```

## Usage
```python
from disturbedyoutubevideosdetection import disturbedyoutubeclassifier as dyc

# Load the Disturbed YouTube Videos Detection Classifier
classifier = dyc.DisturbedYouTubeClassifier(youtube_data_api_key=YOUR_YOUTUBE_DATA_API_KEY)

# Make a prediction
prediction, confidence_score = classifier.predict(video_id=YOUTUBE_VIDEO_ID)
```

## Test the Classifier
You can also test the Disturbed YouTube Videos Detection Classifier using our REST API:
https://api.disturbedyoutubeforkids.xyz/disturbed_youtube_videos_detection?video_id=<YOUTUBE_VIDEO_ID>
