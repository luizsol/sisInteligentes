#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A module to download and parse the CIFAR 10 data.

I modified this module to be PEP-8 and Python3 compliant and to better suit
this project's needs.

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
with 6000 images per class. There are 50000 training images and 10000 test
images.

.. _Original module repository:
    https://github.com/kgeorge/kgeorge_dpl

.. _Current repository:
    https://github.com/luizsol/sisInteligentes

.. _CIFAR 10 Website:
    http://www.cs.toronto.edu/~kriz/cifar.html
"""
__authors__ = ['Koshy George', 'Dongyoung Kim', 'Luiz Sol']
__author__ = 'Luiz Sol'
__version__ = '0.0.1'
__date__ = '2017-10-12'
__maintainer__ = 'Luiz Sol'
__email__ = 'luizedusol@gmail.com'
__status__ = 'Development'

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from random import randint
import re
import sys
import tarfile
import urllib.request


def unpickle(path):
    """Extracts images and labels from CIFAR's files."""
    with open(path, "rb") as f:
        return pickle.load(f, encoding='latin1')


class CifarData:
    """A class to download, extract, load, process and store CIFAR-10 data.

    Constants:
        DATA_URL (str): The Python CIFAR 10 URL.
        DOWNLOADED_DATA_FILEPATH_REG_EXPRESSION (re): A regular expression to
            match the CIFAR 10 downloaded data.
        LOADED_IMG_HEIGHT (int): The CIFAR 10 image's default height in pixels.
        LOADED_IMG_WIDTH (int): The CIFAR 10 image's default width in pixels.
        LOADED_IMG_DEPTH (int): The CIFAR 10 image's default number of channels
            (colors).
        LABELS (list): A list containing the meaning of each image label.

    Attributes:
        train_dataset (dict): A dict containing both the training images and
            labels.
        test_dataset (dict): A dict containing both the test images and labels.

    """
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    # regular expression that matches a datafile
    DOWNLOADED_DATA_FILEPATH_REG_EXPRESSION = re.compile('^data_batch_\d+')
    # cifar-10 consist of 32 x 32 x 3 rgb images
    LOADED_IMG_HEIGHT = 32
    LOADED_IMG_WIDTH = 32
    LOADED_IMG_DEPTH = 3

    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
              'horse', 'ship', 'truck']

    N_CATEGORIES = 10

    def __init__(self, verbose=True, download_and_load=True, data_path=None):
        """The class constructor.

        Keyword Args:
            verbose (bool, default=True): True if this class methods should
                give feedback to the user during it's execution.
            download_and_load (bool, default=True): If True it will download
                CIFAR 10 data in case it isn't present, process and load it.
            data_path (str, default=None): The path to de directory into wich
                the CIFAR 10 data must be stored.

        """
        self.train_dataset = {'data': [], 'labels': [], 'cls': [],
                              'data_array': None, 'labels_matrix': None,
                              'cls_array': None}

        self.test_dataset = {'data': {}, 'labels': [], 'cls': [],
                             'data_array': None, 'labels_matrix': None,
                             'cls_array': None}

        self.verbose = verbose
        self.current_batch_index = 0

        self.shape = (self.LOADED_IMG_HEIGHT, self.LOADED_IMG_HEIGHT,
                      self.LOADED_IMG_DEPTH)

        if download_and_load:
            self.download_and_load(data_path=data_path)

    def download_and_load(self, data_path=None):
        """Downloads (in case it doesn't exists) and load the CIFAR 10 data.

        Keyword Args:
            data_path (str, default=None): The path to de directory into wich
                the CIFAR 10 data will be searched for and stored in case it
                doesn't exists.
        """
        if data_path is None:
            data_path = 'data'

        if not self.check_files(data_path + '/cifar-10-batches-py'):
            self.download_and_extract(data_path=data_path)

        self.load_cifar10_data(data_path=data_path + '/cifar-10-batches-py')

    def check_files(self, data_path):
        """Searches for the CIFAR 10 data into a given directory.

        Args:
            data_path (str): The directory into which the CIFAR data will be
                searched for.
        """
        files = os.listdir(data_path)

        if 'test_batch' not in files:
            return False

        if 'batches.meta' not in files:
            return False

        for i in range(1, 6):
            if 'data_batch_{}'.format(i) not in files:
                return False

        return True

    def download_and_extract(self, data_path=None):
        """Download and extract the tarball from CIFAR 10 website.

        Keyword Args:
            data_path (str, default=None): The path to de directory into wich
                the CIFAR 10 data will be downloaded and extracted.

        """
        if data_path is None:
            data_path = 'data'

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        filename = self.DATA_URL.split('/')[-1]
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename,
                    float(count * block_size) / float(total_size) * 100.0)
                )

                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(CifarData.DATA_URL,
                                                     filepath,
                                                     _progress)
        statinfo = os.stat(filepath)
        self._verbose_print('Successfully downloaded', filename,
                            statinfo.st_size, 'bytes.')

        with tarfile.open(filepath, 'r:gz') as t:
            dataset_dir = os.path.join(data_path, t.getmembers()[0].name)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(t, data_path)

        return dataset_dir

    def show_image(self, image_set='train', index=None, interactive_mode=True):
        """Shows an image stored on this class.

        Keyword Args:
            image_set (str, default='train'): The set from where the image must
                be retrieved (options are 'train' and 'test').
            index (int, default=None): The list index of the image to be
                displayed. If None a random image will be choosed.
            interactive_mode (bool, default=True): If True will activate
                matplotlib.pyplot's interactive mode. If False will disable it.

        """
        if interactive_mode:
            plt.ion()
        else:
            plt.ioff()

        if image_set == 'train':
            target = self.train_dataset
        else:
            target = self.test_dataset

        if index is None:
            index = randint(0, len(target['data']))

        plt.figure(num=self.LABELS[target['labels'][index]])
        plt.imshow(target['data'][index])
        plt.show()

    def load_cifar10_data(self, data_path='data/cifar-10-batches-py',
                          n_train_samples=50000, n_test_samples=10000):
        """Loads CIFAR train and test data.

        The shape of data is 32 x 32 x 3.

        Keyword Args:
            data_path (str, default='data/cifar-10-batches-py'): The directory
                into which the CIFAR data is stored.
            n_train_samples (int, default=50000): The number of training
                samples to be loaded (max=50000).
            n_test_samples (int, default=50000): The number of test
                samples to be loaded (max=10000).

        .. _Original Code:
            https://luckydanny.blogspot.com.br/2016/07/load-cifar-10-dataset-in
                -python3.html

        """
        train_data = None
        train_labels = []

        for i in range(1, 6):
            data_dic = unpickle(data_path + '/data_batch_{}'.format(i))
            if i == 1:
                train_data = data_dic['data']
            else:
                train_data = np.vstack((train_data, data_dic['data']))

            train_labels += data_dic['labels']

        test_data_dic = unpickle(data_path + '/test_batch')
        test_data = test_data_dic['data']
        test_labels = test_data_dic['labels']

        train_data = train_data.reshape((len(train_data),
                                         self.LOADED_IMG_DEPTH,
                                         self.LOADED_IMG_HEIGHT,
                                         self.LOADED_IMG_HEIGHT))

        train_data = np.rollaxis(train_data, 1, 4)
        train_labels = np.array(train_labels)

        test_data = test_data.reshape((len(test_data),
                                       self.LOADED_IMG_DEPTH,
                                       self.LOADED_IMG_HEIGHT,
                                       self.LOADED_IMG_HEIGHT))

        test_data = np.rollaxis(test_data, 1, 4)
        test_labels = np.array(test_labels)

        self.train_dataset = {'data': train_data[0:n_train_samples],
                              'labels': train_labels[0:n_train_samples],
                              'cls': [np.zeros(10)
                                      for i in range(n_train_samples)]}

        for i in range(0, n_train_samples):
            self.train_dataset['cls'][i][self.train_dataset['labels'][i]] = 1.

        self.test_dataset = {'data': test_data[0:n_test_samples],
                             'labels': test_labels[0:n_test_samples],
                             'cls': [np.zeros(10)
                                     for i in range(n_train_samples)]}

        for i in range(0, n_test_samples):
            self.test_dataset['cls'][i][self.test_dataset['labels'][i]] = 1.

        self.train_dataset['data_array'] = np.array(
            [item.flatten() for item in self.train_dataset['data']])

        self.train_dataset['labels_array'] = np.array(
            [item.flatten() for item in self.train_dataset['labels']])

        self.train_dataset['cls_array'] = np.array(
            [item.flatten() for item in self.train_dataset['cls']])

        self.test_dataset['data_array'] = np.array(
            [item.flatten() for item in self.test_dataset['data']])

        self.test_dataset['labels_array'] = np.array(
            [item.flatten() for item in self.test_dataset['labels']])

        self.test_dataset['cls_array'] = np.array(
            [item.flatten() for item in self.test_dataset['cls']])

        return None

    def next_batch(self, batch_size):
        start = self.current_batch_index
        end = start + batch_size

        if end >= len(self.test_dataset['data']):
            start = 0
            end = batch_size

        self.current_batch_index = end

        result_data = self.train_dataset['data_array'][start:end]

        result_labels = self.train_dataset['cls_array'][start:end]

        return (result_data, result_labels)

    def _verbose_print(self, *args):
        if self.verbose:
            print(args)

    def __len__(self):
        return len(self.train_dataset['data'])


if __name__ == "__main__":
    pass
