#
# Copyright 2018 Andrew Lin. All rights reserved.
#
# This software is released under the BSD 3-clause license. See LICENSE.txt or
# https://opensource.org/licenses/BSD-3-Clause for more information.
#
import typing

import numpy as np
import cv2


class MorseVideoThresholdDecoder:
    """Object to decode Morse Code in a video stream using thresholds.

    To use: import this class and call the class method, `decode_morse_video()`
    with the path to the video file.
    """

    # Morse code to alphabet map.
    morse_lookup = {
        '.-': 'A',
        '-...': 'B',
        '-.-.': 'C',
        '-..': 'D',
        '.': 'E',
        '..-.': 'F',
        '--.': 'G',
        '....': 'H',
        '..': 'I',
        '.---': 'J',
        '-.-': 'K',
        '.-..': 'L',
        '--': 'M',
        '-.': 'N',
        '---': 'O',
        '.--.': 'P',
        '--.-': 'Q',
        '.-.': 'R',
        '...': 'S',
        '-': 'T',
        '..-': 'U',
        '...-': 'V',
        '.--': 'W',
        '-..-': 'X',
        '-.--': 'Y',
        '--..': 'Z',
        '.----': '1',
        '..---': '2',
        '...--': '3',
        '....-': '4',
        '.....': '5',
        '-....': '6',
        '--...': '7',
        '---..': '8',
        '----.': '9',
        '-----': '0'
    }

    @staticmethod
    def frame_maxes(path: str) -> typing.Iterator:
        """Generate the max values in each frame of the video file `fname`.
        Assumes the video is grayscale.

        :param path: path to video file.
        :return: iterator of max pixel intensity in each frame of the video.
        """
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ret, frame = cap.read()
                if ret:
                    yield np.max(frame)
                else:
                    break
        finally:
            cap.release()

    @staticmethod
    def signal_runs(
        edges: np.ndarray,
        edges_idx: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Returns number of frames each 'high' and 'low' run are.

        :param edges: numpy array representing edge transitions. +1 for a rising
            edge, -1 for a falling edge, and 0 for no transition.
        :param edges_idx: numpy array of the indexes of the rising and falling
            edges in `edge`
        :return: run lengths.
        """
        runs = edges_idx - np.concatenate(
            (np.array([0], dtype=int), edges_idx[:-1])
        )

        try:
            if edges[edges_idx[0]] > 0:
                # First transition was low to high.
                highs = runs[1::2]
                lows = runs[2::2]
            else:
                # First transition was high to low.
                highs = runs[2::2]
                lows = runs[1::2]
        except IndexError:
            # No edges.
            highs = np.array([], dtype=int)
            lows = np.array([], dtype=int)

        return highs, lows

    @classmethod
    def morse_decode(
        cls,
        edges: np.ndarray,
        edges_idx: np.ndarray,
        dot_dash_boundary: int,
        symbols_word_boundary: int
    ) -> str:
        """Decode Morse Code message in `edges` and `edge_idx`.

        :param edges: numpy array representing edge transitions. +1 for a rising
            edge, -1 for a falling edge, and 0 for no transition.
        :param edges_idx: numpy array of the indexes of the rising and falling
            edges in `edge`
        :param dot_dash_boundary: threshold value between dots and dashes.
        :param symbols_word_boundary: threshold value between symbol silence and
            word silence.
        :return: decoded Morse Code.
        """

        message = []
        morse = []

        try:
            # First full signal is a high if the first edge is a rising edge.
            signal_high = edges[edges_idx[0]] > 0

        except IndexError:
            # No edges.
            pass

        else:
            # Ignore values before leading and trailing edges.
            durations = edges_idx[1:] - edges_idx[:-1]

            for duration in durations.tolist():
                if signal_high:
                    # `duration` is for a 'high' signal (tone).
                    # Convert to 'dot' or 'dash'.
                    morse.append('.' if duration < dot_dash_boundary else '-')
                else:
                    # `duration` is for a 'low' signal (silence).
                    # We do nothing for signal ('.'/'-') boundaries.
                    if duration > dot_dash_boundary:
                        # Symbol boundary. Decode.
                        message.append(cls.morse_lookup.get(''.join(morse), '?'))
                        morse = []  # Reset signal aggregation.
                        if duration > symbols_word_boundary:
                            # Word boundary. Inject word break into message.
                            message.append(' ')

                signal_high = not signal_high

        return ''.join(message)

    @staticmethod
    def compute_threshold(
        signal: np.ndarray,
        tolerance: int
    ) -> int:
        """Return threshold value to split `signal` into two bins.
        Assumes `signal` is bimodal.

        :param signal: signal to threshold.
        :param tolerance: acceptable amount of wiggle when computing threshold.
        :return: computed threshold value.
        """
        # Initial threshold guess is the halfway point.
        threshold = signal.min() + ((signal.max() - signal.min()) / 2)

        # Guarantee the while loop gets entered at least once.
        last_threshold = threshold + tolerance + 1

        # Find the "best" threshold.
        while abs(threshold - last_threshold) > tolerance:
            low = signal[signal < threshold].mean()
            high = signal[signal >= threshold].mean()
            last_threshold = threshold
            threshold = low + ((high - low) / 2)

        return int(threshold)

    @classmethod
    def decode_morse_video(cls, path: str) -> str:
        """Decode Morse Code from video file `path`.

        :param path: path to video file.
        :return: decoded Morse Code.
        """
        # Find maximum pixel magnitude per frame.
        fmax = np.array([m for m in cls.frame_maxes(path)], dtype=int)
        threshold = cls.compute_threshold(fmax, 1)

        # Convert pixel magnitudes to signal on (1) / off (0) values.
        signal = (fmax > threshold).astype(int)

        # Mark transitions.
        #   +1 => rising edge.
        #   -1 => falling edge.
        #   0 => steady state.
        edges = signal[1:] - signal[:-1]

        # Indexes of the edges.
        edges_idx = np.where(edges != 0)[0]

        # Runs between edges, separated into high (signal on) runs and
        # low (signal off) runs.
        highs, lows = cls.signal_runs(edges, edges_idx)

        # Figure out where to threshold dot length, dash length, and word
        # silence length.
        # Assumption: signal break distance is the same as a dot, and symbol
        # break distance is the same as a dash. This is a Morse Code standard.
        dot_dash_threshold = cls.compute_threshold(highs, 1)
        # Filter out the dots to threshold the symbol-word silence threshold.
        symbols_word_threshold = cls.compute_threshold(
            lows[lows > dot_dash_threshold],
            1
        )

        # Decode the message.
        return cls.morse_decode(
            edges,
            edges_idx,
            dot_dash_threshold,
            symbols_word_threshold
        )
