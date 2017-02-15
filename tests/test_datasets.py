import unittest
from util.datasets import get


class TestDatasets(unittest.TestCase):

    def test_get(self):
        self.assertTrue(get('nietzsche') != -1)
        with self.assertRaises(KeyError) as ke:
            get('sup')
        self.assertIsInstance(get('nietzsche'), str)
