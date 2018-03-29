#!/usr/bin/env python3


import tensorflow as tf
from pandas_plink import read_plink
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split


class SingleDataset:
    '''
    Single PLINK formatted dataset for input to the variational autoencoder.
    '''

    def __init__(self, plink_file, scratch_dir, overwrite=False):
        self.options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        self.plink_file = plink_file
        self.scratch_dir = scratch_dir

        # read plink data
        print('Reading PLINK data...')
        self.bim, self.fam, G = read_plink(plink_file)
        # import ipdb; ipdb.set_trace()
        print('Done')

        # write tf.records
        if overwrite:
            G_df = dd.from_dask_array(da.transpose(G))
            G_df = G_df.fillna(value=1) # (. _ . )
            G_df = G_df.astype(np.int8)
            # G_df = G_df.loc[0:200, :]
            tf_records_filenames = G_df.apply(self._write_records, axis=1).compute()
        else:
            root, dirs, files = next(os.walk(scratch_dir))
            tf_records_filenames = [root + f for f in files if f.endswith('.tfrecords')]

        print('Done')

        # split into training and test batches
        self.train_files, self.test_files = train_test_split(tf_records_filenames,
            test_size=0.20, random_state=42)


    def _write_records(self, row):

        sample_j = row.name
        filename = os.path.join(self.scratch_dir,
            os.path.basename(self.plink_file) + '__' + '*'.join(self.fam.loc[self.fam.index[sample_j], ['fid', 'iid']]) + '.tfrecords')

        print('Writing {}...'.format(filename))
        with tf.python_io.TFRecordWriter(filename, options=self.options) as tfwriter:
            genotypes = {'genotypes_raw': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[row.values.tostring()]))}
            example = tf.train.Example(
                features=tf.train.Features(feature=genotypes))
            tfwriter.write(example.SerializeToString())
        return filename


    def decode_tf_records(self, tfrecords_filename):
        '''
        Parse a tf.string pointing to *.tfrecords into a genotype tensor (# variants, # samples)

        Helpful blog post:
        http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
        '''
        data = tf.parse_example([tfrecords_filename],
            {'genotypes_raw': tf.FixedLenFeature([], tf.string)})

        gene_vector = tf.decode_raw(data['genotypes_raw'], np.int8)
        gene_vector = tf.reshape(gene_vector, [self.bim.shape[0]])

        return gene_vector

