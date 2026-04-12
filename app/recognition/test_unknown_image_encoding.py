import unittest

import cv2
import numpy as np

from app.recognition.unknown_embedding_store import ensure_jpeg_bytes


class TestEnsureJpegBytes(unittest.TestCase):
    def test_numpy_image_encodes_to_valid_jpeg(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        jpeg_bytes = ensure_jpeg_bytes(img)

        self.assertIsNotNone(jpeg_bytes)
        self.assertGreater(len(jpeg_bytes), 1024)

        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertGreater(decoded.size, 0)

    def test_tiny_bytes_are_rejected(self):
        self.assertIsNone(ensure_jpeg_bytes(b"abc"))


if __name__ == "__main__":
    unittest.main()
