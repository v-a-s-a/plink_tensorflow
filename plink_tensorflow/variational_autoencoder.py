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

        self.epochs = epochs
        self.latent_dim = latent_dim

        # Data input
        plink_dataset = SingleDataset(plink_file='/plink_tensorflow/data/test/scz_easy-access_wave2.no_trio.bgn',
            scratch_dir='/plink_tensorflow/data/test/',
            overwrite=False)
        self.m_variants = plink_dataset.bim.shape[0]
        self.total_train_batches = (len(plink_dataset.train_files) // batch_size) + 1
        self.total_test_batches = (len(plink_dataset.test_files) // batch_size) + 1

        print('\nTraining Summary:')
        print('\tTraining files: {}'.format(len(plink_dataset.train_files)))
        print('\tTesting  files: {}'.format(len(plink_dataset.test_files)))
        print('\tTraining batches: {}'.format(self.total_train_batches))
        print('\tTesing  batches: {}'.format(self.total_test_batches))

        print('\nBuilding computational graph...')
        # Input pipeline.
        test_dataset = self.build_test_dataset(plink_dataset, batch_size)
        training_dataset = self.build_training_dataset(plink_dataset, batch_size)

        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, training_dataset.output_types, training_dataset.output_shapes)

        self.training_iterator = training_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()

        genotypes = self.iterator.get_next()

        genotypes = tf.cast(genotypes, tf.float32, name='cast_genotypes')
        genotypes.set_shape([None, self.m_variants])

        # Define the model.
        prior = self.make_prior(latent_dim=self.latent_dim)
        make_encoder = tf.make_template('encoder', self.make_encoder)
        posterior = make_encoder(genotypes, latent_dim=self.latent_dim)
        self.latent_z = posterior.sample()

        # Define the loss.
        make_decoder = tf.make_template('decoder', self.make_decoder)

        snp_generator = make_decoder(self.latent_z)
        decoder = tfd.Independent(snp_generator, reinterpreted_batch_ndims=1)

        likelihood = decoder.log_prob(genotypes)
        divergence = tfd.kl_divergence(posterior, prior)
        self.elbo = tf.reduce_mean(likelihood - divergence)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(-self.elbo)

        # export parameters from graph.
        self.probs = make_decoder(prior.sample(1)).probs

        print('Done')


    def build_training_dataset(self, plink_dataset, batch_size):
        with tf.device("/cpu:0"):
            training_dataset = tf.data.TFRecordDataset(plink_dataset.train_files,
                compression_type=tf.constant('ZLIB'))
            training_dataset = training_dataset.shuffle(buffer_size=len(plink_dataset.train_files))
            training_dataset = training_dataset.map(plink_dataset.decode_tf_records)
            training_dataset = training_dataset.batch(batch_size)

        return training_dataset


    def build_test_dataset(self, plink_dataset, batch_size):
        with tf.device("/cpu:0"):
            test_dataset = tf.data.TFRecordDataset(plink_dataset.test_files,
                compression_type=tf.constant('ZLIB'))
            test_dataset = test_dataset.shuffle(buffer_size=len(plink_dataset.test_files))
            test_dataset = test_dataset.map(plink_dataset.decode_tf_records)
            test_dataset = test_dataset.batch(batch_size)

        return test_dataset        


    def infer_parameters(self):
        print('\nExecuting compute graph...')
        # with tf_debug.LocalCLIDebugWrapperSession(tf.Session()) as sess:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_handle = sess.run(self.training_iterator.string_handle())
            test_handle = sess.run(self.test_iterator.string_handle())

            for epoch in tqdm(range(self.epochs), desc='Epoch progress'):
                # train
                sess.run(self.training_iterator.initializer)
                train_pbar = tqdm(total=self.total_train_batches, desc='Train batches')
                for train_batch in range(self.total_train_batches):
                    sess.run(self.optimizer, feed_dict={self.handle: training_handle})
                    train_pbar.update()
                train_pbar.close()

                probs = sess.run(self.probs)
                sample = np.random.binomial(n=2., p=probs)
                print('')
                print(probs)
                print(sample[0, 0:10])

                # test
                sess.run(self.test_iterator.initializer)
                test_elbos = []
                test_pbar = tqdm(total=self.total_test_batches, desc='Test batches')
                for test_batch in range(self.total_test_batches):
                    test_elbo = sess.run(self.elbo, feed_dict={self.handle: test_handle})
                    test_elbos.append(test_elbo)
                    test_pbar.update()
                test_pbar.close()
                print('Epoch;', epoch, 'mean elbo:', np.mean(test_elbos))

        print('Done')


    def make_encoder(self, data, latent_dim):
        x = tf.layers.dense(data, 200, tf.nn.relu)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        x = tf.layers.dropout(x, 0.1)
        loc = tf.layers.dense(x, latent_dim)
        scale = tf.layers.dense(x, latent_dim, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)


    def make_prior(self, latent_dim):
        loc = tf.zeros(latent_dim)
        scale = tf.ones(latent_dim)
        return tfd.MultivariateNormalDiag(loc, scale)


    def make_decoder(self, z):
        x = tf.layers.dense(z, 200, tf.nn.relu)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        logits = tf.layers.dense(x, self.m_variants)
        logits = tf.reshape(logits, [-1] + [self.m_variants])
        return tfd.Binomial(logits=logits, total_count=2.)


if __name__ == '__main__':
    vae = BasicVariationalAutoencoder()
    vae.infer_parameters()
