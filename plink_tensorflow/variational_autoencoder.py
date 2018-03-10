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

from single_dataset import SingleDataset


class BasicVariationalAutoencoder():

    def __init__(self, batch_size = 100, latent_dim = 2, epochs = 50):
        
        # Data input
        self.input_dataset = SingleDataset(plink_file='/plink_tensorflow/data/hapmap1',
            scratch_dir='/plink_tensorflow/data/')

        print('Building computational graph.')
        self.filenames = tf.placeholder(tf.string, shape=[], name='tf_records_filename')
        self.dataset = tf.data.TFRecordDataset(self.filenames, compression_type=tf.constant('ZLIB'))
        self.dataset = self.dataset.map(self.input_dataset.decode_tf_records)
        
        self.iterator = self.dataset.make_initializable_iterator()
        print('Done')

        # self.dataset = tf.data.TFRecordDataset.concatenate(self.dataset)
        # shape = self.dataset.shape
        # TODO: merge the tensors from individual studies
        # TODO: calculate correct batch size
        # self.dataset = self.dataset.batch(10)
        # iterator = self.dataset.make_initializable_iterator()
        # next_element = iterator.get_next()

        # Define the model.
        # prior = self._make_prior(latent_dim=2)
        # make_encoder = tf.make_template('encoder', self._make_encoder) 
        # posterior = make_encoder(self.dataset, latent_dim=2)
        # self.latent_z = posterior.sample()

        # # Define the loss.
        # make_decoder = tf.make_template('decoder', self._make_decoder)
        # likelihood = make_decoder(self.latent_z,
        #     [self.input_dataset.m_variants]).log_prob(next_element)
        # divergence = tfd.kl_divergence(posterior, prior)
        # self.elbo = tf.reduce_mean(likelihood - divergence)
        # self.optimizer = tf.train.AdamOptimizer(0.001).minimize(-self.elbo)


    def infer_parameters(self):
        # with tf_debug.LocalCLIDebugWrapperSession(tf.train.MonitoredSession()) as sess:
        print('Executing compute graph.')
        with tf.Session() as sess:


            # sess.run(self.iterator.initializer,
            #     feed_dict={self.filenames: self.input_dataset.plink_file})
            sess.run(self.iterator.initializer, feed_dict={self.filenames: self.input_dataset.plink_file + '.tfrecords'})
            print(sess.run(self.iterator.get_next()))

            # for epoch in tqdm(range(50)):
            #     self.input_dataset.test_train_split()
            #     test_feed = {self.data: self.input_dataset.test_set()}
            #     test_elbo, test_codes = sess.run([self.elbo, self.latent_z], test_feed)
            #     print('Epoch', epoch, 'elbo', test_elbo)
            #     for training_batch in tqdm(self.input_dataset.train_set_minibatches()):
            #         train_feed = {self.data: training_batch}
            #         sess.run(self.optimizer, train_feed)
        
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


    def _make_decoder(self, z, data_shape):
        x = tf.layers.dense(z, 200, tf.nn.relu)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        logit = tf.layers.dense(x)
        logit = tf.reshape(logit, [-1] + logit.shape)
        return tfd.Independent(tfd.Binomial(logits=logit, total_count=2.))


if __name__ == '__main__':
    vae = BasicVariationalAutoencoder()
    vae.infer_parameters()
