import logging
import sys
sys.path.append("..")
import os
import unittest
import numpy as np
from utils.io_utils import *
from data import TestData

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

class TestIOUtils(unittest.TestCase):

    def setUp(self):
        self.batch_size = 16
        self.num_batches = 10
        self.num_residual = 2
        self.vocab_size = 50

    def test_batch_generator(self):
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger('TestIOUtils.test_scope')
        self.dataset = TestData()

        log.info("Testing batch_concatenate on TestData.")
        data = self.dataset.train_data
        self.assertIsInstance(data, tuple)
        self.assertTrue(len(data) == 2)
        self.assertIsInstance(data[0], list)
        self.assertIsInstance(data[1], list)

        # Create random test data.
        # plus 1 for residual.
        max_seq_lengths = np.random.randint(20, 100, size=self.num_batches+1)
        encoder_sentences = []
        decoder_sentences = []
        for b in range(self.num_batches):
            encoder_batch = np.random.randint(self.vocab_size, size=(self.batch_size, max_seq_lengths[b]))
            encoder_sentences.append(encoder_batch)
            decoder_batch = np.random.randint(self.vocab_size, size=(self.batch_size, max_seq_lengths[b]))
            decoder_sentences.append(decoder_batch)
        encoder_batch = np.random.randint(self.vocab_size, size=(self.num_residual, max_seq_lengths[-1]))
        encoder_sentences.append(encoder_batch)
        decoder_batch = np.random.randint(self.vocab_size, size=(self.num_residual, max_seq_lengths[-1]))
        decoder_sentences.append(decoder_batch)

        self.assertEqual(len(encoder_sentences), len(decoder_sentences))
        gen = batch_generator(encoder_sentences, decoder_sentences, self.batch_size)

        batch_ctr = 0
        for batch_gen in gen:
            self.assertEqual(batch_gen[0].shape, batch_gen[1].shape)
            log.info("Batch counter: %d" % batch_ctr)
            log.info("\tencoder batch: %r" % batch_gen[0])
            log.info("\tdecoder batch: %r" % batch_gen[1])
            batch_ctr += 1




if __name__ == '__main__':
    unittest.main()
