# Video Morse Decoder

An analysis of the Morse Code message from a video recording of the new 
signalling tower located on the Revelle Campus of the University of California, 
San Diego, and the subsequent Python application derived from it.

## Use

To decode a (small, grayscale) video of a signal light flashing Morse Code,
enter the following on the command line:

```$ python -m decode_morse_video.py <path to video>```

To interact with my Jupyter Notebook on decoding a video of a signal light
flashing Morse Code, create an appropriate conda environment (see Requirements,
below), and run Jupyter Notebook from this directory.

```$ jupyter notebook```

## Requirements

The Morse video decoder requires Python, OpenCV, and a bunch of Python 
libraries. By far, the easiest way to obtain Python and OpenCV (at least
on Linux) is to install [Anaconda](https://anaconda.com), and make a conda 
virtual environment with the necessary libraries, like so:

```$ conda create -n morsedecoder --file conda-requirements.txt```

`ffmpeg` is also useful, but not required, to get your video into a digestible
format (grayscale, and as small as possible) from the command line.

## Video Conversion

Some helpful `ffmpeg` commands:

* Resize, and strip audio:
  `ffmpeg -i <source> -s <width>x<height> -an <destination>`
* Convert to grayscale:
  `ffmpeg -i <source> -vf format=gray <destination>`
* Crop video:
  `ffmpeg -i <source> -vf crop=<width>:<height>:<x>:<y> <destination>`

You may or may not be able to successfully combine commands.

## License

Copyright 2018, Andrew Lin. All rights reserved.

This software is released under the BSD 3-clause license. See LICENSE.txt or 
https://opensource.org/licenses/BSD-3-Clause for more information.
