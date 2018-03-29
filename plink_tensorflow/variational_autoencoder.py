#!/usr/bin/env python3
'''
Based on the excellent blog post by Danijar Hafner:
https://danijar.com/building-variational-auto-encoders-in-tensorflow/
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from tqdm import tqdm
from tensorflow.python import debug as tf_debug
from sklearn.model_selection import train_test_split

from datasets import SingleDataset


class BasicVariationalAutoencoder():

    def __init__(self, batch_size = 1000, latent_dim = 25, epochs = 50):

        # Data input
        self.input_dataset = SingleDataset(plink_file='/plink_tensorflow/data/test/scz_easy-access_wave2.no_trio.bgn',
            scratch_dir='/plink_tensorflow/data/test/',
            overwrite=False)
        self.m_variants = self.input_dataset.bim.shape[0]
        self.latent_dim = latent_dim

        print('Building computational graph.')
        self.training_filenames = tf.placeholder(tf.string, shape=[None])
        self.test_filenames = tf.placeholder(tf.string, shape=[None])

        self.training_dataset = tf.data.TFRecordDataset(self.training_filenames, compression_type=tf.constant('ZLIB'))
        self.training_dataset = self.training_dataset.map(self.input_dataset.decode_tf_records)
        self.training_dataset = self.training_dataset.batch(batch_size)

        self.test_dataset = tf.data.TFRecordDataset(self.test_filenames, compression_type=tf.constant('ZLIB'))
        self.test_dataset = self.test_dataset.map(self.input_dataset.decode_tf_records)
        self.test_dataset = self.test_dataset.batch(batch_size)

        self.iterator = self.dataset.make_initializable_iterator()
        genotypes = self.iterator.get_next()

        genotypes = tf.cast(genotypes, tf.float32, name='cast_genotypes')
        genotypes.set_shape([None, self.m_variants])

        print('Done')

        # Define the model.
        prior = self._make_prior(latent_dim=self.latent_dim)
        make_encoder = tf.make_template('encoder', self._make_encoder) 
        posterior = make_encoder(genotypes, latent_dim=self.latent_dim)
        self.latent_z = posterior.sample()

        # Define the loss.
        make_decoder = tf.make_template('decoder', self._make_decoder)
        likelihood = make_decoder(self.latent_z).log_prob(genotypes)
        divergence = tfd.kl_divergence(posterior, prior)
        self.elbo = tf.reduce_mean(likelihood - divergence)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(-self.elbo)


    def infer_parameters(self):
        print('Executing compute graph.')
        with tf_debug.LocalCLIDebugWrapperSession(tf.Session()) as sess:
        # with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in tqdm(range(50)):
                # test set
                sess.run(self.iterator.initializer,
                    feed_dict={self.filenames: self.input_dataset.test_files})
                while True:
                    # consume batches
                    try:
                        test_elbo = sess.run(self.elbo,
                            feed_dict={self.filenames: self.input_dataset.train_files})
                        print('Epoch', epoch, 'elbo', test_elbo)
                    except tf.errors.OutOfRangeError:
                        break

                # training set
                sess.run(self.iterator.initializer,
                    feed_dict={self.filenames: self.input_dataset.train_files})
                while True:
                    # consume batches
                    try:
                        sess.run(self.optimizer)
                    except tf.errors.OutOfRangeError:
                        break

        print('Done')


    def _make_encoder(self, data, latent_dim):
        x = tf.layers.dense(data, 200, tf.nn.relu)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        loc = tf.layers.dense(x, latent_dim)
        scale = tf.layers.dense(x, latent_dim, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)


    def _make_prior(self, latent_dim):
        loc = tf.zeros(latent_dim)
        scale = tf.ones(latent_dim)
        return tfd.MultivariateNormalDiag(loc, scale)


    def _make_decoder(self, z):
        x = tf.layers.dense(z, 200, tf.nn.relu)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        logits = tf.layers.dense(x, self.m_variants)
        return tfd.Independent(tfd.Binomial(logits=logits, total_count=2.), reinterpreted_batch_ndims=1)


if __name__ == '__main__':
    vae = BasicVariationalAutoencoder()
    vae.infer_parameters()
