#!/usr/bin/env python3


import tensorflow as tf
from pandas_plink import read_plink
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


from sklearn.model_selection import train_test_split


class SingleDataset:
    '''
    Single PLINK formatted dataset for input to the variational autoencoder.
    '''

    def __init__(self, plink_file, scratch_dir, overwrite=False, seed=42):
        self.options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        self.plink_file = plink_file
        self.scratch_dir = scratch_dir

        # read plink data
        print('\nReading PLINK data...')
        self.bim, self.fam, G = read_plink(plink_file)

        ## some testing
        # self.fam = self.fam.loc[0:9, :]
        # self.bim = self.bim.loc[0:20, :]
        # surrogate = np.full((20, 10), 1.)
        # surrogate[1, :] = 2.0 * np.ones(10)
        # surrogate[5, :] = np.zeros(10)
        # surrogate[6, :] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # G = da.from_array(surrogate.T, chunks=1)

        self.m_variants = G.shape[0]

        print('Done')

        if overwrite:
            # split into training and test batches
            train_samples, test_samples = train_test_split(self.fam,
                test_size=0.20, random_state=seed)
            self.train_size = train_samples.shape[0]
            self.test_size = test_samples.shape[0]

            # write tf.records
            plink_base = os.path.basename(self.plink_file)
            self.test_filename = os.path.join(self.scratch_dir,
                     '{}.{}-sample.test.tfrecords'.format(plink_base,
                        test_samples.shape[0]))
            self.train_filename = os.path.join(self.scratch_dir,
                     '{}.{}-sample.train.tfrecords'.format(plink_base,
                        train_samples.shape[0]))


            G_df = dd.from_dask_array(da.transpose(G))
            G_df = G_df.fillna(value=1) # (. _ . )
            G_df = G_df.astype(np.int8)

            with tf.python_io.TFRecordWriter(self.test_filename, options=self.options) as tfwriter:
                test_df = G_df.loc[test_samples.index.values, :]
                test_write_pbar = tqdm(total=self.train_size, desc='Writing Test tfRecords')
                for i, row in test_df.iterrows():
                    self._write_records(row, tfwriter)
                    test_write_pbar.update()
                test_write_pbar.close()

            train_write_pbar = tqdm(total=self.train_size, desc='Writing Train tfRecords')
            with tf.python_io.TFRecordWriter(self.train_filename, options=self.options) as tfwriter:
                train_df = G_df.loc[train_samples.index.values, :]
                for i, row in train_df.iterrows():
                    self._write_records(row, tfwriter)
                    train_write_pbar.update()
                train_write_pbar.close()

        else:
            root, dirs, files = next(os.walk(scratch_dir))
            tfrecords = [root+f for f in files if f.endswith('.tfrecords')]
            (self.test_filename,) = [f for f in tfrecords if 'test' in os.path.basename(f)]
            (self.train_filename,) = [f for f in tfrecords if 'train' in os.path.basename(f)]
            self.test_size = int(os.path.basename(self.test_filename).split('.')[3].replace('-sample', ''))
            self.train_size = int(os.path.basename(self.train_filename).split('.')[3].replace('-sample', ''))


    def _write_records(self, row, writer_handle):
        '''
        Write a sample's genotype vectors to a file.
        '''
        genotypes_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[row.values.tostring()]))

        # convert to Example
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'genotypes': genotypes_feature}))

        writer_handle.write(example.SerializeToString())


    def decode_tfrecords(self, example_proto):
        '''
        Parse a tf.string pointing to *.tfrecords into a genotype tensor (# variants, # samples)

        Helpful blog post:
        http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

        Helpful repo:
        https://github.com/visipedia/tfrecords/blob/master/create_tfrecords.py
        '''
        feature_map =  {'genotypes': tf.FixedLenFeature([], tf.string)}
        data = tf.parse_single_example(example_proto, feature_map)

        genotypes = tf.decode_raw(data['genotypes'], np.int8)
        genotypes = tf.reshape(genotypes, [self.m_variants])
        genotypes = tf.cast(genotypes, tf.float32)

        return genotypes

