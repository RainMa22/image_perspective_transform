import unittest

import numpy as np

from perspective_transformer import euclidean_distance;


class testPerspectiveTransform(unittest.TestCase):
    def test_euclidean_distance(self):
        v1 = np.float32([3, 4])
        self.assertEqual(euclidean_distance(v1), 5.)  # add assertion here


if __name__ == '__main__':
    unittest.main()
