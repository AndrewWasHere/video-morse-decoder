#
# Copyright 2018 Andrew Lin. All rights reserved.
#
# This software is released under the BSD 3-clause license. See LICENSE.txt or
# https://opensource.org/licenses/BSD-3-Clause for more information.
#
import numpy as np
import pytest

from morse_video_decoder.morse_video_threshold_decoder import MorseVideoThresholdDecoder


def test_empty():
    """Test `compute_threshold` with empty array."""
    with pytest.raises(ValueError):
         MorseVideoThresholdDecoder.compute_threshold(
             np.array([], dtype=int),
             1
         )


def test_impulses():
    """Test `compute_threshold` with bimodal impulses."""
    t = MorseVideoThresholdDecoder.compute_threshold(
        np.array([0, 4], dtype=int),
        1
    )

    assert t == 2


def test_distributions():
    """Test `compute_threshold` with bimodal distributions."""
    t = MorseVideoThresholdDecoder.compute_threshold(
        np.array([1, 2, 3, 4, 11, 12, 12, 12], dtype=int),
        1
    )

    assert t == 7
