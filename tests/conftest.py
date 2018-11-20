#
# Copyright 2018 Andrew Lin. All rights reserved.
#
# This software is released under the BSD 3-clause license. See LICENSE.txt or
# https://opensource.org/licenses/BSD-3-Clause for more information.
#
import numpy as np
import pytest


@pytest.fixture
def extract_edges():
    """Edge extraction function."""
    def f(signal: np.ndarray):
        edges = signal[1:] - signal[:-1]
        edges_idx = np.where(edges != 0)[0]

        return edges, edges_idx

    return f
