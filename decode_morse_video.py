#
# Copyright 2018, Andrew Lin. All rights reserved.
#
# This software is released under the BSD 3-clause license. See LICENSE.txt or
# https://opensource.org/licenses/BSD-3-Clause for more information.
#
import argparse
import os

from morse_video_decoder.morse_video_threshold_decoder import \
    MorseVideoThresholdDecoder


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        help='path to grayscale video file to decode. Keep it small!'
    )
    args = parser.parse_args()
    args.path = os.path.abspath(os.path.expanduser(args.path))

    return args


def main():
    args = parse_command_line()
    print(MorseVideoThresholdDecoder.decode_morse_video(args.path))


if __name__ == '__main__':
    main()
