#!/usr/bin/env python
'''
Based on the excellent blog post by Danijar Hafner:
https://danijar.com/building-variational-auto-encoders-in-tensorflow/
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm_notebook


import tensorflow.contrib.distributions as tfd


def main():

    # input data
    ## TODO: replace with tf.flags
    # latent_df = pd.read_csv('./data/spatial_a-0.1_N-10000_M-1000.latent_rep.txt', sep='\t')
    # log_dir = ''
    # X = np.load('../data/spatial_a-0.1_N-10000_M-1000.pandas.gz.npy')
    # X = X.astype(np.float32)
    # X_train, X_test, latent_df_train, latent_df_test = train_test_split(X, latent_df, test_size=0.2)


    M_variants = X.shape[1]
    N_samples = X.shape[0]

    batch_size = 100
    latent_dim = 2
    epochs = 50

    # define out model
    graph = tf.Graph()
    with graph.as_default():
    
        def make_encoder(data, latent_dim):
            x = tf.layers.flatten(data)
            x = tf.layers.dense(x, 200, tf.nn.relu)
            x = tf.layers.dense(x, 200, tf.nn.relu)
            loc = tf.layers.dense(x, latent_dim)
            scale = tf.layers.dense(x, latent_dim, tf.nn.softplus)
            return tfd.MultivariateNormalDiag(loc, scale)


        def make_prior(latent_dim):
            loc = tf.zeros(latent_dim)
            scale = tf.ones(latent_dim)
            return tfd.MultivariateNormalDiag(loc, scale)


        def make_decoder(z, data_shape):
            x = z
            x = tf.layers.dense(x, 200, tf.nn.relu)
            x = tf.layers.dense(x, 200, tf.nn.relu)
            logit = tf.layers.dense(x, np.prod(data_shape))
            logit = tf.reshape(logit, [-1] + data_shape)
            return tfd.Independent(tfd.Binomial(logits=logit, total_count=2.))
        
        
        # Data input
        data = tf.placeholder(tf.float32, shape=[None, M_variants])    

        # Define the model.
        prior = make_prior(latent_dim=2)
        make_encoder = tf.make_template('encoder', make_encoder) # tf scoping
        posterior = make_encoder(data, latent_dim=2)
        code = posterior.sample()

        # Define the loss.
        make_decoder = tf.make_template('decoder', make_decoder) # tf scoping
        likelihood = make_decoder(code, [M_variants]).log_prob(data)
        divergence = tfd.kl_divergence(posterior, prior)
        elbo = tf.reduce_mean(likelihood - divergence)
        optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)

        # Infer parameters.
        saver = tf.train.Saver()
        with tf.train.MonitoredSession() as sess:

            for epoch in tqdm_notebook(range(50)):
                test_feed = {data: X_test}
                test_elbo, test_codes = sess.run([elbo, code], test_feed)
                print('Epoch', epoch, 'elbo', test_elbo)
                for train_batch in tqdm_notebook(minibatch(X_train, batch_size=10), leave=False):
                    train_feed = {data: X_train}
                    sess.run(optimize, train_feed)
            
            saver.save(sess, FLAGS.train_dir, global_step)


if __name__ == '__main__':
    main()
