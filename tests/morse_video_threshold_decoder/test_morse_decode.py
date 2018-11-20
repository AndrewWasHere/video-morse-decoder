#
# Copyright 2018 Andrew Lin. All rights reserved.
#
# This software is released under the BSD 3-clause license. See LICENSE.txt or
# https://opensource.org/licenses/BSD-3-Clause for more information.
#
import numpy as np

from morse_video_decoder.morse_video_threshold_decoder import \
    MorseVideoThresholdDecoder


def test_decode_empty(extract_edges):
    """`morse_decode` of an empty array should return an empty string. """
    edges = np.array([], dtype=int)
    edges_idx = np.array([], dtype=int)

    s = MorseVideoThresholdDecoder.morse_decode(edges, edges_idx, 2, 7)

    assert s == ''


def test_decode_no_transitions():
    """`morse_decode` an array with no transitions."""
    edges = np.zeros(10, dtype=int)
    edges_idx = np.array([], dtype=int)

    s = MorseVideoThresholdDecoder.morse_decode(edges, edges_idx, 2, 7)

    assert s == ''


def test_decode_one_transition(extract_edges):
    """`morse_decode` an array with only one transition."""
    signal = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=int)
    edges, edges_idx = extract_edges(signal)

    s = MorseVideoThresholdDecoder.morse_decode(edges, edges_idx, 2, 7)

    assert s == ''


def test_so(extract_edges):
    """`morse_decode` SO."""
    signal = np.array(
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        dtype=int
    )
    edges, edges_idx = extract_edges(signal)

    s = MorseVideoThresholdDecoder.morse_decode(edges, edges_idx, 2, 7)

    assert s == 'SO'
