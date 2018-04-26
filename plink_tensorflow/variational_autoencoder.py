#!/usr/bin/env python3
'''
Based on the excellent blog post by Danijar Hafner:
https://danijar.com/building-variational-auto-encoders-in-tensorflow/
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from tqdm import tqdm
from tensorflow.python import debug as tf_debug
from sklearn.model_selection import train_test_split

from tensorflow.contrib.nn import conv1d_transpose


from datasets import SingleDataset


class ConvolutionalVariationalAutoencoder():

    def __init__(self, batch_size = 1000, latent_dim = 25, epochs = 50, log_dir='/plink_tensorflow/experiments/'):

        self.log_dir = log_dir
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        # Data input
        plink_dataset = SingleDataset(plink_file='/plink_tensorflow/data/test/scz_easy-access_wave2.no_trio.bgn',
            scratch_dir='/plink_tensorflow/data/test/',
            overwrite=False)
        self.m_variants = plink_dataset.m_variants
        self.total_train_batches = (plink_dataset.train_size // batch_size) + 1
        self.total_test_batches = (plink_dataset.test_size // batch_size) + 1
    
        self.bim = plink_dataset.bim

        print('\nTraining Summary:')
        print('\tTraining samples: {}'.format(plink_dataset.train_size))
        print('\tTesting  samples: {}'.format(plink_dataset.test_size))
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
        self.genotypes = genotypes

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
            training_dataset = tf.data.TFRecordDataset([plink_dataset.train_filename],
                compression_type=tf.constant('ZLIB'))
            training_dataset = training_dataset.map(plink_dataset.decode_tfrecords)
            training_dataset = training_dataset.shuffle(buffer_size=plink_dataset.train_size)
            training_dataset = training_dataset.batch(batch_size)

        return training_dataset


    def build_test_dataset(self, plink_dataset, batch_size):
        with tf.device("/cpu:0"):
            test_dataset = tf.data.TFRecordDataset([plink_dataset.test_filename],
                compression_type=tf.constant('ZLIB'))
            test_dataset = test_dataset.map(plink_dataset.decode_tfrecords)
            test_dataset = test_dataset.shuffle(buffer_size=plink_dataset.test_size)
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

                # test
                sess.run(self.test_iterator.initializer)
                test_elbos = []
                test_pbar = tqdm(total=self.total_test_batches, desc='Test batches')
                for test_batch in range(self.total_test_batches):
                    test_elbo = sess.run(self.elbo, feed_dict={self.handle: test_handle})
                    test_elbos.append(test_elbo)
                    test_pbar.update()
                test_pbar.close()
                print('Epoch', epoch, ': mean elbo =', np.mean(test_elbos))
                
                probs = self.probs.eval()
                self.bim['probs_' + str(epoch)] = probs[0]
                self.bim.to_csv(os.path.join(self.log_dir, 'mhc_bim.tsv'), sep='\t')

        print('Done')


    # Create a function that creates one-dimensional convolutional layers
    ## taken from: https://bigdatascientistblog.wordpress.com/2016/12/09/tensorflow-201-beyond-hello-world-mnist/
    def conv1d_relu(self, layer_name, input, kernel_shape, bias_shape, stride):
        with tf.variable_scope(layer_name):
            weights = tf.get_variable('weights', kernel_shape,
                initializer=tf.random_normal_initializer())
            biases = tf.get_variable('biases', bias_shape,
                initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv1d(input, weights,
                stride=stride, padding='VALID')
    
            return tf.nn.relu(conv + biases)
    

    def make_encoder(self, data, latent_dim):
        '''
        Gene vector -> latent variable

        Some basics on 1D convolutions in tf:
        https://stackoverflow.com/questions/38114534/basic-1d-convolution-in-tensorflow?rq=1
        '''
        x = tf.reshape(data, [self.batch_size, self.m_variants, 1])

        print('Encoder layer sizes:')
        print(x.get_shape().as_list())
        x = tf.layers.conv1d(x, 32, 5, strides=1, activation=tf.nn.relu, padding='SAME')
        print(x.get_shape().as_list())

        x = tf.layers.conv1d(x, 64, 5, strides=4, activation=tf.nn.relu, padding='SAME')
        print(x.get_shape().as_list())

        x = tf.layers.conv1d(x, 128, 5, strides=4, activation=tf.nn.relu, padding='VALID')
        print(x.get_shape().as_list())

        # drop channel dimension
        x = tf.reshape(x, [self.batch_size, -1])

        x = tf.layers.dense(x, self.latent_dim * 2, activation=None)
        
        loc = x[:, :self.latent_dim]
        scale = tf.nn.softplus(x[:, self.latent_dim:])

        print(loc.get_shape().as_list())
        print(scale.get_shape().as_list())

        return tfd.MultivariateNormalDiag(loc, scale)


    def make_prior(self, latent_dim):
        loc = tf.zeros(latent_dim)
        scale = tf.ones(latent_dim)
        return tfd.MultivariateNormalDiag(loc, scale)


    def make_decoder(self, z):
        '''
        Latent variable -> gene vector

        This was helpful:
        https://analysiscenter.github.io/dataset/_modules/dataset/models/tf/layers/conv.html#conv1d_transpose
        '''

        # add channel and synthetic dimension to sequence
        x = tf.reshape(z, [self.batch_size, 1, self.latent_dim, 1])

        print('Decoder layer sizes:')
        print(x.get_shape().as_list())
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=(1, 20),
            strides=(1, 1), padding='VALID', activation=tf.nn.relu)
        x = tf.squeeze(x, [1])
        print(x.get_shape().as_list())

        x = tf.expand_dims(x, axis=1)
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=(1, 10),
            strides=(1, 1), padding='VALID', activation=tf.nn.relu)
        x = tf.squeeze(x, [1])

        print(x.get_shape().as_list())
        x = tf.expand_dims(x, axis=1)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=(1, 5),
            strides=(1, 4), padding='SAME', activation=tf.nn.relu)
        x = tf.squeeze(x, [1])

        x = tf.expand_dims(x, axis=1)
        x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=(1, 5),
            strides=(1, 4), padding='SAME', activation=tf.nn.relu)
        x = tf.squeeze(x, [1])
        print(x.get_shape().as_list())

        logits = tf.reshape(x, [self.batch_size, -1])

        return tfd.Binomial(logits=logits, total_count=2.)


if __name__ == '__main__':
    vae = ConvolutionalVariationalAutoencoder()
    vae.infer_parameters()
