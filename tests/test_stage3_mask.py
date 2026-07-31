import pytest

torch = pytest.importorskip("torch")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.stage3_cleanup import refine_bubble_mask  # noqa: E402


def test_refine_bubble_mask_finds_white_circle():
    img = np.full((200, 200), 128, np.uint8)
    cv2.circle(img, (100, 100), 30, 255, -1)
    mask = refine_bubble_mask(img, (60, 60, 80, 80))
    assert mask[100, 100] > 0
    assert mask[5, 5] == 0
