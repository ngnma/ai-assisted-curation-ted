import numpy as np

from ted_curation.seed import set_seed


def test_same_seed_gives_same_numbers():
    set_seed(1)
    a = np.random.rand(5)
    set_seed(1)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
