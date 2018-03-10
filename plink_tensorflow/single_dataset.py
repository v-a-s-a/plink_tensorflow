#!/usr/bin/env python3


import tensorflow as tf
from pandas_plink import read_plink
import dask.array as da
import numpy as np
import os


class SingleDataset:
    '''
    Single PLINK formatted dataset for input to the variational autoencoder.
    '''

    def __init__(self, plink_file, scratch_dir):
        self.options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        self.plink_file = plink_file

        # read plink data
        print('Reading PLINK data...')
        self.bim, self.fam, G = read_plink(plink_file)
        print('Done')

        # fill missing values
        G_df = G.to_dask_dataframe()
        G = G_df.fillna(axis=0, method='backfill').values.compute().astype(np.int8) # (. _ . )

        # write tf.records
        filename = os.path.join(scratch_dir, os.path.basename(plink_file) + '.tfrecords')
        print('Writing {}...'.format(filename))
        with tf.python_io.TFRecordWriter(filename, options=self.options) as tfwriter:
            # write each individual gene vector to record
            genotypes = {'genotypes_raw': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=G.flatten())),
                         'sample_size': tf.train.Feature(int64=G.shape[1]),
                         'num_variants': tf.train.Feature(int64=G.shape[1])}
            example = tf.train.Example(
                features=tf.train.Features(feature=genotypes))
            tfwriter.write(example.SerializeToString())
        print('Done')


    def decode_tf_records(self, tfrecords_filename):
        '''
        Helpful blog post:
        http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
        '''
        parsed_features = tf.parse_example(tfrecords_filename,
            {'gene_matrix': tf.FixedLenFeature((), tf.int64)})
        return parsed_features['gene_vector']
    