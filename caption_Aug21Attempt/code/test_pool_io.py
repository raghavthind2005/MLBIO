"""Tests for image-cell decoding.

The bug these exist for (job 3167490) was a schema assumption carried from a
different dataset: `images` was treated as a list of image structs, but
Vision-SR1-47K declares a singular `Image` feature, so the cell IS the struct and
`cell[0]` raised `KeyError: 0`. Cheap to catch here, 55 s of GPU to catch there.
"""

from __future__ import annotations

import unittest

from pool_io import extract_image_bytes

PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


class TestExtractImageBytes(unittest.TestCase):
    def test_singular_image_struct(self):
        """Vision-SR1-47K's actual shape."""
        self.assertEqual(extract_image_bytes({"bytes": PNG, "path": "a.png"}), PNG)

    def test_list_of_one_struct(self):
        """ViRL39K's shape -- still supported so the loader is dataset-portable."""
        self.assertEqual(extract_image_bytes([{"bytes": PNG, "path": "a.png"}]), PNG)

    def test_raw_bytes(self):
        self.assertEqual(extract_image_bytes(PNG), PNG)

    def test_multi_image_row_is_a_hard_error_not_a_silent_first_pick(self):
        """R8: one image per item. Silently taking [0] would give `c` an
        undefined referent while every log line still looked healthy."""
        with self.assertRaises(AssertionError) as cm:
            extract_image_bytes([{"bytes": PNG}, {"bytes": PNG}])
        self.assertIn("R8", str(cm.exception))

    def test_empty_list_is_an_error(self):
        with self.assertRaises(AssertionError):
            extract_image_bytes([])

    def test_struct_without_bytes_names_its_keys(self):
        with self.assertRaises(AssertionError) as cm:
            extract_image_bytes({"path": "a.png"})
        self.assertIn("path", str(cm.exception))

    def test_unrecognised_type_is_an_error(self):
        with self.assertRaises(AssertionError):
            extract_image_bytes(42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
