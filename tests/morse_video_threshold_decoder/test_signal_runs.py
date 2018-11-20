#
# Copyright 2018 Andrew Lin. All rights reserved.
#
# This software is released under the BSD 3-clause license. See LICENSE.txt or
# https://opensource.org/licenses/BSD-3-Clause for more information.
#
import numpy as np

from morse_video_decoder.morse_video_threshold_decoder import \
    MorseVideoThresholdDecoder


def test_empty():
    """Test `signal_runs` with empty arrays."""
    edges = np.array([], dtype=int)
    edges_idx = np.array([], dtype=int)

    highs, lows = MorseVideoThresholdDecoder.signal_runs(edges, edges_idx)

    assert highs.size == lows.size == 0


def test_ignore_leading_and_trailing_state(extract_edges):
    """Leading and trailing states should be ignored because the start or end of
    the state is unknown.
    """
    signal = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=int)
    edges, edges_idx = extract_edges(signal)

    highs, lows = MorseVideoThresholdDecoder.signal_runs(edges, edges_idx)

    assert highs.size == lows.size == 0


def test_multiple_transitions(extract_edges):
    """Test `signal_runs` with multiple transitions."""
    signal = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=int)
    edges, edges_idx = extract_edges(signal)

    highs, lows = MorseVideoThresholdDecoder.signal_runs(edges, edges_idx)

    assert np.array_equal(highs, np.array([3, 1], dtype=int))
    assert np.array_equal(lows, np.array([1, 3], dtype=int))
