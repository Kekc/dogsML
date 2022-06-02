import numpy as np
import os

import dogsml.settings
import dogsml.utils.dataset


def test_extract_image_data_from_path():
    test_path = os.path.join(
        dogsml.settings.TEST_ROOT,
        "test_data/dog_0000.jpg"
    )

    data = dogsml.utils.dataset.extract_image_data_from_path(test_path)

    assert isinstance(data, np.ndarray)
    assert data.shape == (64, 64, 3)

    flattened = data.reshape(-1)
    assert all(list(flattened >= 0))
    assert all(list(flattened <= 1))

    data = dogsml.utils.dataset.extract_image_data_from_path(
        test_path,
        width=100,
        height=50,
    )

    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 100, 3)
