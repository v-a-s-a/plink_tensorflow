#!/usr/bin/env python3


import tensorflow as tf
from pandas_plink import read_plink
import dask.array as da
import numpy as np
import os

from sklearn.model_selection import train_test_split


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
        tf_records_filenames = []
        for sample_j in range(G.shape[1]):
            filename = os.path.join(scratch_dir,
                os.path.basename(plink_file) + '__' + '*'.join(self.fam.loc[self.fam.index[sample_j], ['fid', 'iid']]) + '.tfrecords')
            # print('Writing {}...'.format(filename))
            # with tf.python_io.TFRecordWriter(filename, options=self.options) as tfwriter:
            #     # write each individual gene vector to record
            #     genotypes = {'genotypes_raw': tf.train.Feature(
            #                     bytes_list=tf.train.BytesList(value=[G[:, sample_j].tobytes()]))}
            #     example = tf.train.Example(
            #         features=tf.train.Features(feature=genotypes))
            #     tfwriter.write(example.SerializeToString())
            tf_records_filenames.append(filename)
        # print('Done')

        # split into training and test batches
        self.train_files, self.test_files = train_test_split(tf_records_filenames,
            test_size=0.20, random_state=42)


    def decode_tf_records(self, tfrecords_filename):
        '''
        Parse a tf.string pointing to *.tfrecords into a genotype tensor (# variants, # samples)

        Helpful blog post:
        http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
        '''
        data = tf.parse_example([tfrecords_filename],
            {'genotypes_raw': tf.FixedLenFeature([], tf.string)})

        gene_vector = tf.decode_raw(data['genotypes_raw'], tf.int8)
        gene_vector = tf.reshape(gene_vector, tf.stack([1, self.bim.shape[0]]))

        return gene_vector

