#
# Created on Sat May 20 2023
#
# Copyright (c) 2023 Colin
#
from typing import Any, List
import os
import pickle

from absl import flags
import numpy as np
import tensorflow as tf
import tree

FLAGS = flags.FLAGS


def _nested_stack(sequence: List[Any]):
    """Stack nested elements in a sequence."""
    return tree.map_structure(lambda *x: np.stack(x), *sequence)


class DemonstrationRecorder:
    """Records demonstrations.

    A demonstration is a (observation, action, reward, discount) tuple where
    every element is a numpy array corresponding to a full episode.
    """

    def __init__(self):
        self._demos = []
        self._reset_episode()

    def step(self, observation: np.ndarray, action: np.ndarray,
             reward: np.ndarray = None, discount: np.ndarray = None):
        reward = reward or np.array(0, np.float32)
        self._episode_reward += reward
        self._episode.append((observation, action, reward,
                              discount or np.array(1.0, np.float32)))

    def record_episode(self):
        self._demos.append(_nested_stack(self._episode))
        self._reset_episode()

    def discard_episode(self):
        self._reset_episode()

    def _reset_episode(self):
        self._episode = []
        self._episode_reward = 0

    @property
    def episode_reward(self):
        return self._episode_reward

    def save(self, folder: str, filename: str = 'demos.pkl'):
        os.makedirs(folder, exist_ok=True)
        # pickle save demos
        with open(os.path.join(folder, filename), 'wb') as f:
            pickle.dump(self._demos, f)

    def load(self, folder: str, filename: str = 'demos.pkl'):
        # pickle load demos
        with open(os.path.join(folder, filename), 'rb') as f:
            self._demos = pickle.load(f)
        return self

    def make_tf_dataset(self):
        types = (tf.float32, tf.float32, tf.float32, tf.float32)
        shapes = tree.map_structure(lambda x: x.shape, self._demos[0])
        ds = tf.data.Dataset.from_generator(lambda: self._demos, types, shapes)
        return ds.repeat().shuffle(len(self._demos))
