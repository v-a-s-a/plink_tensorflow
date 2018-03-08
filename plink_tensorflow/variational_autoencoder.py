#!/usr/bin/env python
'''
Based on the excellent blog post by Danijar Hafner:
https://danijar.com/building-variational-auto-encoders-in-tensorflow/
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

from plink_feed import MetaAnalysisDataset


class BasicVariationalAutoencoder():

    def __init__(self):

        # FLAGS
        batch_size = 100
        latent_dim = 2
        epochs = 50
        
        # Data input
        input_dataset = MetaAnalysisDataset(test_prop=0.5)
        data = tf.placeholder(tf.float32, shape=[None, input_dataset.m_variants])

        # Define the model.
        prior = self._make_prior(latent_dim=2)
        make_encoder = tf.make_template('encoder', self._make_encoder) # tf scoping
        posterior = make_encoder(data, latent_dim=2)
        code = posterior.sample()

        # Define the loss.
        make_decoder = tf.make_template('decoder', self._make_decoder) # tf scoping
        likelihood = make_decoder(code, [input_dataset.m_variants]).log_prob(data)
        divergence = tfd.kl_divergence(posterior, prior)
        elbo = tf.reduce_mean(likelihood - divergence)
        optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)

        # Infer parameters.
        saver = tf.train.Saver()
        with tf.train.MonitoredSession() as sess:
            for epoch in range(50):
                input_dataset.test_train_split()
                test_feed = {data: input_dataset.test_set()}
                test_elbo, test_codes = sess.run([elbo, code], test_feed)
                print('Epoch', epoch, 'elbo', test_elbo)
                for training_batch in input_dataset.train_set_minibatches():
                    train_feed = {data: training_batch}
                    sess.run(optimize, train_feed)
            
            saver.save(sess, FLAGS.train_dir, global_step)

    def _make_encoder(self, data, latent_dim):
        x = tf.layers.flatten(data)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        loc = tf.layers.dense(x, latent_dim)
        scale = tf.layers.dense(x, latent_dim, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)


    def _make_prior(self, latent_dim):
        loc = tf.zeros(latent_dim)
        scale = tf.ones(latent_dim)
        return tfd.MultivariateNormalDiag(loc, scale)


    def _make_decoder(self, z, data_shape):
        x = z
        x = tf.layers.dense(x, 200, tf.nn.relu)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        logit = tf.layers.dense(x, np.prod(data_shape))
        logit = tf.reshape(logit, [-1] + data_shape)
        return tfd.Independent(tfd.Binomial(logits=logit, total_count=2.))


if __name__ == '__main__':
    vae = BasicVariationalAutoencoder()    
