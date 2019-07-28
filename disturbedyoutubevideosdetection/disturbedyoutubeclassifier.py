# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 Kostantinos Papadamou, Cyprus University of Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from socket import error as SocketError
import numpy as np
from sklearn import preprocessing
import isodate as isodate
import emoji
import os
import re
import requests
import shutil
from . import disturbedyoutubemodelfeatureextractor


class DisturbedYouTubeClassifier(object):
    def __init__(self, youtube_data_api_key):
        """
        C'tor
        """
        # Disturbed YouTube Videos Detection Classifier details
        self.disturbed_youtube_classifier_path = 'disturbed_youtube_videos_detection_model.hdf5'
        # Set Classes Labels
        self.classes = ['appropriate', 'disturbing']

        # Define Base Directories
        self.VIDEO_DATA_DIR = 'video_data/'

        def PRECISION(y_true, y_pred):
            """
            Precision metric.
            Only computes a batch-wise average of precision.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def RECALL(y_true, y_pred):
            """
            Recall metric.
            Only computes a batch-wise average of recall.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def F1_SCORE(y_true, y_pred):
            """
            F1 Score metric.
            Only computes a batch-wise average of recall.
            """
            p = PRECISION(y_true, y_pred)
            r = RECALL(y_true, y_pred)
            return (2 * p * r) / (p + r + K.epsilon())

        def AUC_SCORE(y_true, y_pred):
            """
            ROC_AUC Score metric.
            Only computes a batch-wise average of roc_auc.
            """
            auc = tf.metrics.auc(y_true, y_pred, curve='ROC')[1]
            K.get_session().run(tf.local_variables_initializer())
            return auc

        # Load the Disturbed YouTube Videos Detection Classifier
        self.disturbed_youtube_classifier = load_model(filepath=self.disturbed_youtube_classifier_path,
                                                       custom_objects={
                                                           'PRECISION': PRECISION,
                                                           'RECALL': RECALL,
                                                           'F1_SCORE': F1_SCORE,
                                                           'AUC_SCORE': AUC_SCORE})

        # YouTube API Configuration for getting video details
        self.YOUTUBE_API_SERVICE_NAME = "youtube"
        self.YOUTUBE_API_VERSION = "v3"

        # Get YouTube Data API key
        self.YOUTUBE_API_KEY = youtube_data_api_key

        # Create a YouTUbe Data API Instance
        self.YOUTUBE_DATA_API = build(serviceName=self.YOUTUBE_API_SERVICE_NAME,
                                      version=self.YOUTUBE_API_VERSION,
                                      developerKey=self.YOUTUBE_API_KEY)

        # Create a CNN Feature Extractor Object
        self.CNN_feature_extractor_model = disturbedyoutubemodelfeatureextractor.CNN_MODEL()

    def get_video_information(self, video_id):
        """
        Method that queries the YouTube Data API to retrieve the details of a given video_id.
        :param video_id: the video_id to download from YouTube
        :param related_videos: the list of the related videos of the provided video_id
        :return: a dictionary with the downloaded video's information
        """
        # Send HTTP Request to get Video Info
        while True:
            # Send request to get video's information
            try:
                response = self.YOUTUBE_DATA_API.videos().list(
                    part='id, snippet, statistics',
                    id=video_id
                ).execute()

                # Get Video's information from the response
                try:
                    video_information = response['items'][0]
                except:
                    return {}

                return video_information
            except (HttpError, SocketError) as error:
                print('[ERROR] HTTP Error occurred while retrieving information of Video ID: %s.' % (video_id))

    @staticmethod
    def key_exists(element, *keys):
        """
        Method that checks if a (nested) key exists in a dict element
        :param element: the dictionary to be checked
        :param keys: the keys to check if exist in the provided dictionary
        :return: TRUE if the provided keys exist in the dictionary, otherwise FALSE
        """
        if type(element) is not dict:
            raise AttributeError('keys_exists() expects dict as first argument.')
        if len(keys) == 0:
            raise AttributeError('keys_exists() expects at least two arguments, one given.')

        _element = element
        for key in keys:
            try:
                _element = _element[key]
            except KeyError:
                return False

        return True

    @staticmethod
    def get_num_of_emoticons(str):
        """
        Method that calculates the number of emoticons in the given text
        :param str: the string to be checked
        :return: the number of emoticons in the given string
        """
        num_emoticons = 0

        for character in str:
            if character in emoji.UNICODE_EMOJI:
                num_emoticons += 1

        return num_emoticons

    @staticmethod
    def get_jaccard_similarity(str1, str2):
        """
        Method that calculates the Jaccard Similarity between two strings
        :param str1: the first string
        :param str2: the second string
        :return: the jaccard simillarity of the two provided strings
        """
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        jaccard_sim = float(len(c)) / (len(a) + len(b) - len(c))

        return float("{0:.2f}".format(jaccard_sim))

    def YTDurationToSeconds(self, duration):
        """
        Method that converts a YouTube video's duration to seconds
        :param duration: the duration of the video
        :return: the duration of the video in seconds
        """
        match = re.match('PT(\d+H)?(\d+M)?(\d+S)?', duration).groups()
        hours = self._js_parseInt(match[0]) if match[0] else 0
        minutes = self._js_parseInt(match[1]) if match[1] else 0
        seconds = self._js_parseInt(match[2]) if match[2] else 0
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def _js_parseInt(string):
        """
        JavaScript-like parseInt method
        :param string: the integer like string
        :return: the Integer value of the provided string
        """
        return int(''.join([x for x in string if x.isdigit()]))

    def get_video_general_style_features(self, video_information):
        """
        Method that extracts the style (other- and statistics-related) features for a given YouTube Video's Information
        :param video_information: the dictionary with all the video information to be used
        :return: a numpy-like array with all the video's style features
        """
        # init variables
        video_general_style_features = list()
        bad_words = ['sex', 'undress', 'kiss', 'kill', 'killed', 'smoke', 'weed', 'burn', 'die', 'dead', 'death', 'burried',
                     'alive', 'suicide', 'poop', 'inject', 'injection', 'arrested', 'hurt', 'naked', 'blood', 'bloody']

        kids_related_words = ['tiaras', 'kid', 'kids', 'toddler', 'toddlers', 'surprise', 'fun', 'funny', 'disney',
                              'school', 'learn', 'superheroes', 'heroes', 'family', 'baby', 'mickey']

        """
        Get Video-related Features
        """
        # Get Video's duration in seconds
        try:
            duration = video_information['contentDetails']['duration']
            video_duration = isodate.parse_duration(duration)
            video_duration_in_seconds = video_duration.total_seconds()
        except KeyError:
            video_duration_in_seconds = 0
        # Add it to the result
        video_general_style_features.append(int(video_duration_in_seconds))

        # Get Video's Category
        try:
            video_categoryId = int(video_information['snippet']['categoryId'])
        except KeyError:
            video_categoryId = 0
        video_general_style_features.append(video_categoryId)

        """
        Get Video Statistics-related Features
        """
        # get video views
        try:
            video_views_cntr = int(video_information['statistics']['viewCount'])
        except KeyError:
            video_views_cntr = 0
        # get video likes
        try:
            video_likes_cntr = int(video_information['statistics']['likeCount'])
        except KeyError:
            video_likes_cntr = 0
        # get video dislikes
        try:
            video_dislikes_cntr = int(video_information['statistics']['dislikeCount'])
        except KeyError:
            video_dislikes_cntr = 0
        if video_dislikes_cntr > 0:
            video_likes_dislikes_ratio = int(video_likes_cntr / video_dislikes_cntr)
        else:
            video_likes_dislikes_ratio = video_likes_cntr
        try:
            video_comments_cntr = int(video_information['statistics']['commentCount'])
        except KeyError:
            video_comments_cntr = 0

        # Add all of them to the result
        video_general_style_features.append(video_views_cntr)
        video_general_style_features.append(video_likes_cntr)
        video_general_style_features.append(video_dislikes_cntr)
        video_general_style_features.append(video_likes_dislikes_ratio)
        # video_general_style_features.append(video_added_favourites_cntr)
        video_general_style_features.append(video_comments_cntr)


        """
        Get Video Title- and description-related Features
        """
        # get video title and split it into words
        video_title = video_information['snippet']['title']
        words_in_video_title = video_title.split()
        # get video description and split it into words
        video_description = video_information['snippet']['description']
        words_in_video_description = video_description.split()

        # get title length
        video_title_length = len(words_in_video_title)
        # get description length
        video_description_length = len(words_in_video_description)
        # get description ratio over the title
        video_description_title_ratio = int(video_description_length / video_title_length)
        # get jaccard similarity between words appearing in title and description
        video_description_title_jaccard_similarity = self.get_jaccard_similarity(video_description, video_title)

        # get number of exclamation and question marks in title
        video_title_exclamation_marks_cntr = video_title.count('!')
        video_title_question_marks_cntr = video_title.count('?')
        # get number of emoticons in title
        video_title_emoticons_cntr = self.get_num_of_emoticons(video_title)
        # get number of bad words in title
        video_title_bad_words_cntr = 0
        for word in words_in_video_title:
            if word.lower() in bad_words:
                video_title_bad_words_cntr += 1
        # get number of kids-related words in title
        video_title_kids_related_words_cntr = 0
        for word in words_in_video_title:
            if word.lower() in kids_related_words:
                video_title_kids_related_words_cntr += 1

        # get number of exclamation and question marks in description
        video_description_exclamation_marks_cntr = video_description.count('!')
        video_description_question_marks_cntr = video_description.count('?')
        # get number of emoticons in description
        video_description_emoticons_cntr = self.get_num_of_emoticons(video_description)
        # get number of bad words in description
        video_description_bad_words_cntr = 0
        for word in words_in_video_description:
            if word.lower() in bad_words:
                video_description_bad_words_cntr += 1
        # get number of kids-related words in title
        video_description_kids_related_words_cntr = 0
        for word in words_in_video_description:
            if word.lower() in kids_related_words:
                video_description_kids_related_words_cntr += 1

        # Add all of them to the result
        video_general_style_features.append(video_title_length)
        video_general_style_features.append(video_description_length)
        video_general_style_features.append(video_description_title_ratio)
        video_general_style_features.append(video_description_title_jaccard_similarity)

        video_general_style_features.append(video_title_exclamation_marks_cntr)
        video_general_style_features.append(video_title_question_marks_cntr)
        video_general_style_features.append(video_title_emoticons_cntr)
        video_general_style_features.append(video_title_bad_words_cntr)
        video_general_style_features.append(video_title_kids_related_words_cntr)

        video_general_style_features.append(video_description_exclamation_marks_cntr)
        video_general_style_features.append(video_description_question_marks_cntr)
        video_general_style_features.append(video_description_emoticons_cntr)
        video_general_style_features.append(video_description_bad_words_cntr)
        video_general_style_features.append(video_description_kids_related_words_cntr)


        """
        Get Video Tags-related Features
        """
        try:
            video_tags = video_information['snippet']['tags']
        except KeyError:
            video_tags = []

        # get number of tags in the video
        video_tags_cntr = len(video_tags)
        # init other features variables in case this video has no tags
        video_tags_bad_words_cntr = 0
        video_tags_kids_related_words_cntr = 0
        video_tags_title_jaccard_similarity = 0

        if video_tags_cntr > 0:
            # get number of bad words in video tags
            for tag in video_tags:
                # check if there is more than 1 word in the current video tag
                if ' ' in tag:
                    words_in_current_tag = tag.split()
                    for word in words_in_current_tag:
                        if word.lower() in bad_words:
                            video_tags_bad_words_cntr += 1
                else:
                    if tag.lower() in bad_words:
                        video_tags_bad_words_cntr += 1

            # get number of kids-related words in video tags
            for tag in video_tags:
                # check if there is more than 1 word in the current video tag
                if ' ' in tag:
                    words_in_current_tag = tag.split()
                    for word in words_in_current_tag:
                        if word.lower() in bad_words:
                            video_tags_kids_related_words_cntr += 1
                else:
                    if tag.lower() in bad_words:
                        video_tags_kids_related_words_cntr += 1

            # get Jaccard similarity between words appearing in video tags and description
            # create a text string from all the tags
            video_tags_text = ''
            for tag in video_tags:
                video_tags_text += tag + ' '
            video_tags_title_jaccard_similarity = self.get_jaccard_similarity(video_tags_text, video_title)

        # Add everything to the result
        video_general_style_features.append(video_tags_cntr)
        video_general_style_features.append(video_tags_bad_words_cntr)
        video_general_style_features.append(video_tags_kids_related_words_cntr)
        video_general_style_features.append(video_tags_title_jaccard_similarity)

        """
        Convert the result array to a numpy array
        """
        final_video_general_style_features = np.expand_dims(np.asarray(video_general_style_features), axis=0)
        final_video_general_style_features = preprocessing.normalize(final_video_general_style_features, axis=0).astype("float64")

        return final_video_general_style_features

    def preprocess_headlines_one_hot(self, video_headline, vocab_size=10277, headline_max_words_length=21):
        """
        Method that extracts the features of a given video headline
        :param video_headline: the video's Headline
        :param vocab_size:
        :param headline_max_words_length: the bigger headline size from all the headlines used to train the classifier
        :return: the preprocessed one-hot encoded headline
        """

        # Integer encode the document
        encoded_headline = one_hot(video_headline, vocab_size)

        # Perform padding
        headline_features = sequence.pad_sequences(np.expand_dims(encoded_headline, axis=0), maxlen=headline_max_words_length)

        return np.array(headline_features)

    def preprocess_video_tags_one_hot(self, video_tags, vocab_size=19040, video_tags_max_length=95):
        """
        Method that performs the required pre-processing for the video tags of a given YouTube Video ID
        using keras.preprocessing.text.one_hot. It is actually a bag-of-words technique.
        :param video_tags: the list with all the video tags to be processed
        :param vocab_size: the vocabulary size of all the videos' video tags used to train the classifier
        :param video_tags_max_length: the bigger video tags size from all the headlines used to train the classifier
        :return: the preprocessed one-hot encoded video_tags numpy array
        """
        # init variables
        final_encoded_video_tags, encoded_video_tags = list(), list()

        if len(video_tags) == 0:
            # this video has not tags so add an empty array
            final_encoded_video_tags.append([])
        else:
            # encode each tag of the video separately
            for tag in video_tags:
                encoded_video_tags += one_hot(tag, vocab_size)

            # append the current video's encoded video tags array
            final_encoded_video_tags.append(encoded_video_tags)

        # Perform padding
        video_tags_features = sequence.pad_sequences(final_encoded_video_tags, maxlen=video_tags_max_length)

        return np.array(video_tags_features)

    def download_thumbnail(self, url, video_id):
        """
        Download thumbnail from the given URL found in the returned Video's JSON Data
        :param url: the url of the thumbnail to be downloaded
        :param video_id: the video_id
        :return: TRUE if the thumbnail has been downloaded successfully, otherwise FALSE
        """
        try:
            # Download Thumbnail
            response = requests.get(url, stream=True, timeout=10)

            # Check if the video's thumbnail directory exists and if not create it
            original_umask = os.umask(0)
            try:
                if not os.path.exists(self.THUMBNAIL_DIR + video_id):
                    os.makedirs(self.THUMBNAIL_DIR + video_id, 0o777)
            finally:
                os.umask(original_umask)

            # Create Thumbnail filename
            extension = url.split('/')[-1].split('.')[-1]
            filename = video_id + '.' + extension

            # Store the thumbnail file but first check if we already downloaded it
            to_download = self.THUMBNAIL_DIR + video_id + '/' + filename
            if not os.path.exists(to_download):
                if response.status_code == 200:
                    with open(to_download, 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                return True
        except Exception as e:
            print(str(e))
            return False

    def delete_video_downloaded_data(self, video_id):
        """
        Method that deletes all Video's downloaded data (thumbnail, etc.)
        :param video_id: the video_id to use and delete downloaded information
        """
        # Delete the folder with all the downloaded information
        if os.path.exists(self.VIDEO_DATA_DIR + video_id):
            os.system("rm -R " + self.VIDEO_DATA_DIR + video_id)
        return

    def predict(self, video_id):
        """
        Method that receives a YouTube Video ID and uses the Disturbed YouTube classifier to predict
        the correct class for that video.
        :param video_id: the YouTube video_id that we will make a prediction
        :return: the predicted class, otherwise UNKNOWN
        """
        # Download Video Information
        video_information = self.get_video_information(video_id=video_id)

        if not video_information:
            return 'UNKNOWN'

        """
        Get Video's Style Features (General Features also used for Training the Basic ML Models)
        """
        X_general_style_features_in = self.get_video_general_style_features(video_information=video_information)

        """
        Get Headline Features
        """
        video_headline = video_information['snippet']['title']
        X_headline_features_in = self.preprocess_headlines_one_hot(video_headline=video_headline)

        """
        Get Video Tags Features
        """
        if self.key_exists(video_information, 'snippet', 'tags'):
            video_tags = video_information['snippet']['tags']
        else:
            video_tags = list()
        X_video_tags_features_in = self.preprocess_video_tags_one_hot(video_tags=video_tags)

        """
        Get Video Thumbnail Features
        """
        try:
            # Get Thumbnail URL
            video_thumbnail_url = video_information['snippet']['thumbnails']['high']['url']
            # Check if we have already downloaded that thumbnail and if NOT then Download it
            if not os.path.isfile(self.VIDEO_DATA_DIR + video_id + '.jpg'):
                thumbnail_downloaded = self.download_thumbnail(url=video_thumbnail_url,
                                                               video_id=video_id)
            else:
                thumbnail_downloaded = True

            if thumbnail_downloaded:
                thumbnail_features = self.CNN_feature_extractor_model.extract_features_image(frame_image_path=self.VIDEO_DATA_DIR + video_id + '.jpg')
            else:
                thumbnail_features = self.CNN_feature_extractor_model.extract_features_image(frame_image_path='',
                                                                                             isEmptyImage=True)
        except KeyError:
            thumbnail_features = self.CNN_feature_extractor_model.extract_features_image(frame_image_path='',
                                                                                         isEmptyImage=True)

        """
        Make Prediction
        """
        prediction_proba = self.disturbed_youtube_classifier.predict([np.expand_dims(thumbnail_features, axis=0),
                                                                      X_headline_features_in,
                                                                      X_general_style_features_in,
                                                                      X_video_tags_features_in],
                                                                     batch_size=1)

        """
        Decode predicted probabilities to get the actual predicted label
        """
        # Convert probability to class offset
        prediction = prediction_proba.argmax(axis=-1)
        # Get predicted class from offset
        predicted_class = self.classes[prediction[0]]

        """
        Delete video's downloaded data
        """
        self.delete_video_downloaded_data(video_id=video_id)

        return predicted_class, prediction_proba