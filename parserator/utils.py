#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys

if  sys.version_info[0] == 2:
    reload_func = reload
elif sys.version_info <= (3, 3):
    import imp
    reload_func = imp.reload
else:
    import importlib
    reload_func = importlib.reload

from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator
from .training import readTrainingData
import pycrfsuite


def f1_with_flattening(estimator, X, y):
    """
    Calculate F1 score by flattening the predictions of the
    estimator across all sequences. For example, given the following
    address sequences as input
        ['1 My str', '2 Your blvd'],
    the predictions of the model will be flattened like so:
        ['AddressNumber', 'StreetName', 'StreetNamePostType', 'AddressNumber', 'StreetName', 'StreetNamePostType']
    and compared to a similarly flattened gold standard labels. This calculates the overall
    quality of the model across all sequences as opposed to how well it does
    at any particular sequence.
    :param X: list of sequences to tag
    :param y: list of gold standard tuples
    """
    predicted = estimator.predict(X)
    flat_pred, flat_gold = [], []
    for a, b in zip(predicted, y):
        if len(a) == len(b):
            flat_pred.extend(a)
            flat_gold.extend(b)
    return f1_score(flat_gold, flat_pred)


def get_data_sklearn_format(train_file_list, module):
    """
    Parses the specified data files and returns it in sklearn format.
    :param path:
    :return: tuple of:
                1) list of training sequences, each of which is a string
                2) list of gold standard labels, each of which is a tuple
                of strings, one for each token in the corresponding training
                sequence
    """
    data = list(readTrainingData(train_file_list, module.GROUP_LABEL))
    random.shuffle(data)

    x, y = [], []
    for raw_string, components in data:
        tokens, labels = zip(*components)
        x.append(raw_string)
        y.append(labels)
    return x, y


class SequenceEstimator(BaseEstimator):
    """
    A sklearn-compatible wrapper for a parser trainer
    """

    def __init__(self, parserator_module, model_path=None, c1=1, c2=1, feature_minfreq=0):
        """
        :param c1: L1 regularisation coefficient
        :param c2: L2 regularisation coefficient
        :param feature_minfreq: minimum feature frequency
        :return:
        """
        self.parserator_module = parserator_module
        self.model_path = model_path
        self.c1 = c1
        self.c2 = c2
        self.feature_minfreq = feature_minfreq

    def fit(self, X, y, **params):
        # sklearn requires parameters to be declared as fields of the estimator,
        # an we can't have a full stop there. Replace with an underscore
        params = {k.replace('_', '.'): v for k, v in self.__dict__.items()}
        trainer = pycrfsuite.Trainer(verbose=False, params=params)
        for raw_text, labels in zip(X, y):
            tokens = self.parserator_module.tokenize(raw_text)
            trainer.append(self.parserator_module.tokens2features(tokens), labels)
        trainer.train(self.model_path)
        reload_func(self.parserator_module)

    def predict(self, X):
        reload_func(self.parserator_module)  # tagger object is defined at the module level, update now
        predictions = []
        for sequence in X:
            predictions.append([foo[1] for foo in self.parserator_module.parse(sequence)])
        return predictions


