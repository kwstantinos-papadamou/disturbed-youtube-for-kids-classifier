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

import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model


class CNN_MODEL(object):
    def __init__(self):
        """
        C'tor
        """

        # Set image input shape
        self.image_input_shape_channels = (299, 299, 3)
        self.image_input_shape = (299, 299)

        # Get CNN model with pre-trained ImageNet weights. Include also the top fully connected layer
        # and we will freeze the last softmax classification layer in the next line
        CNN_model = InceptionV3(weights='imagenet',
                                include_top=True)

        # Freeze the last softmax layer. We'll extract features at the final pool layer.
        self.model = Model(inputs=CNN_model.input,
                           outputs=CNN_model.get_layer('avg_pool').output)

    def extract_features_image(self, frame_image_path, isEmptyImage=False):
        """
        Method that takes as input the path of an image (in our case a video THUMBNAIL)
        converts it to a numpy array using Keras.preprocessing package and then extracts the features from that
        image frame by running it into the pre-trained CNN Inception V3 model
        :param frame_image_path: the path to get the image file
        :param isEmptyImage: True if there is no thumbnail file, otherwise False
        :return: the thumbnail image features
        """
        if not isEmptyImage:
            # Read image
            img = image.load_img(frame_image_path, target_size=self.image_input_shape) # only requires wxh size and adds channels by default
            x = image.img_to_array(img) # 299x299x3
            x = np.expand_dims(x, axis=0) # 1x299x299x3
            x = preprocess_input(x) # 1x299x299x3
        else:
            # Create an image zero numpy array of size (299x299x3)
            img = np.zeros(self.image_input_shape_channels)
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        features = features[0]

        return features