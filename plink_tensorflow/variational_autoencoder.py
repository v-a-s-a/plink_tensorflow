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

from single_dataset import SingleDataset


class BasicVariationalAutoencoder():

    def __init__(self, batch_size = 10, latent_dim = 25, epochs = 50):

        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        # Data input
        self.input_dataset = SingleDataset(plink_file='/plink_tensorflow/data/hapmap1',
            scratch_dir='/plink_tensorflow/data/')
        self.m_variants = self.input_dataset.bim.shape[0]

        print('Building computational graph.')
        self.filenames = tf.placeholder(tf.string, shape=[None], name='tf_records_filename')
        self.dataset = tf.data.TFRecordDataset(self.filenames, compression_type=tf.constant('ZLIB'))
        self.dataset = self.dataset.map(self.input_dataset.decode_tf_records)
        # self.dataset = self.dataset.repeat(epochs)
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        self.iterator = self.dataset.make_initializable_iterator()
        genotypes = self.iterator.get_next()

        genotypes = tf.cast(genotypes, tf.float32)
        genotypes.set_shape([self.batch_size, self.m_variants])

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
        # with tf_debug.LocalCLIDebugWrapperSession(tf.train.MonitoredSession()) as sess:
        print('Executing compute graph.')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in tqdm(range(self.epochs)):
                # test set
                sess.run(self.iterator.initializer,
                    feed_dict={self.filenames: self.input_dataset.test_files})
                while True:
                    # consume batches
                    try:
                        test_elbo, test_codes = sess.run([self.elbo, self.latent_z])
                        print('Epoch', epoch, 'elbo', test_elbo)
                    except tf.errors.OutOfRangeError:
                        print('something ')
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
        '''
        Gene vector -> latent variable

        Some basics on 1D convolutions in tf:
        https://stackoverflow.com/questions/38114534/basic-1d-convolution-in-tensorflow?rq=1
        '''
        x = tf.reshape(data, [self.batch_size, self.m_variants, 1])

        x = tf.layers.conv1d(x, 32, 5, strides=4, activation=tf.nn.relu, padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)

        x = tf.layers.conv1d(x, 64, 5, strides=4, activation=tf.nn.relu, padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)

        x = tf.layers.conv1d(x, 128, 5, strides=4, activation=tf.nn.relu, padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)

        x = tf.layers.conv1d(x, 256, 3, activation=tf.nn.relu, padding='VALID')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)

        x = tf.layers.dropout(x, rate=0.1)

        x = tf.reshape(x, [self.batch_size, -1])
        x = tf.layers.dense(x, self.latent_dim * 2, activation=None)
        loc = x[:, :self.latent_dim]
        scale = tf.nn.softplus(x[:, self.latent_dim:])

        # loc = tf.layers.dense(x, latent_dim, activation=None)
        # scale = tf.layers.dense(x, latent_dim, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)


    def _make_prior(self, latent_dim):
        loc = tf.zeros(latent_dim)
        scale = tf.ones(latent_dim)
        return tfd.MultivariateNormalDiag(loc, scale)

    def _make_decoder(self, z):
        '''
        Latent variable -> gene vector
        '''
        x = tf.expand_dims(z, axis=-1)
        x = tf.expand_dims(x, axis=-1)
         
        x = tf.layers.conv2d_transpose(x, 256, 3, strides=2, padding='VALID')

        print(x.get_shape())

        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)

        x = tf.layers.conv2d_transpose(x, 128, 5, strides=4, padding='VALID')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)

        x = tf.layers.conv2d_transpose(x, 64, 5, strides=4, padding='VALID')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)

        x = tf.layers.conv2d_transpose(x, 32, 5, strides=4, padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.elu(x)
        
        logits = tf.layers.conv2d_transpose(x, 1, 5, strides=4, padding='SAME')
        logits = tf.squeeze(logits, axis=-1)
        logits = tf.squeeze(logits, axis=2)

        return tfd.Independent(tfd.Binomial(logits=logits, total_count=2.))


if __name__ == '__main__':
    vae = BasicVariationalAutoencoder()
    vae.infer_parameters()
