#
# Copyright 2018 Andrew Lin. All rights reserved.
#
# This software is released under the BSD 3-clause license. See LICENSE.txt or
# https://opensource.org/licenses/BSD-3-Clause for more information.
#
import random
from unittest import mock

import numpy as np

from morse_video_decoder.morse_video_threshold_decoder import \
    MorseVideoThresholdDecoder


def generate_test_data():
    max_frame = random.randint(0, 42)
    frame = np.array(
        [max_frame - random.randint(0, max_frame) for _ in range(10)]
    )
    idx = random.randint(0, len(frame) - 1)
    frame[idx] = max_frame
    return max_frame, frame


def test_no_frames():
    """Test `frame_maxes` when there are no frames of video."""
    mock_read = mock.MagicMock(return_value=(False, None))
    mock_capture = mock.MagicMock(read=mock_read)
    with mock.patch('cv2.VideoCapture', return_value=mock_capture):
        frames = [f for f in MorseVideoThresholdDecoder.frame_maxes('path')]

    assert len(frames) == 0


def test_single_frame():
    """Test `frame_maxes` for the proper number of frames and values yielded."""
    max_frame, frame = generate_test_data()
    mock_read = mock.MagicMock(
        side_effect=[
            (True, frame),
            (False, None)
        ]
    )
    mock_capture = mock.MagicMock(read=mock_read)
    with mock.patch('cv2.VideoCapture', return_value=mock_capture):
        frames = [f for f in MorseVideoThresholdDecoder.frame_maxes('path')]

    assert len(frames) == 1
    assert frames[0] == max_frame


def test_multiple_frames():
    """Test `frame_maxes` for the proper number of frames and values yielded."""
    max_frames, test_frames = zip(*[generate_test_data() for _ in range(10)])
    yields = [(True, f) for f in test_frames]
    yields.append((False, None))
    mock_read = mock.MagicMock(side_effect=yields)
    mock_capture = mock.MagicMock(read=mock_read)
    with mock.patch('cv2.VideoCapture', return_value=mock_capture):
        frames = [f for f in MorseVideoThresholdDecoder.frame_maxes('path')]

    assert tuple(frames) == max_frames
