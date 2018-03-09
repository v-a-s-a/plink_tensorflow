#!/usr/bin/env python3
'''
Make sure our data has not been corrupted when writing to tf.record.
'''

import tensorflow as tf

from plink_feed import MetaAnalysisDataset


def decode_plink_tf_record(filename):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    record_iterator = tf.python_io.tf_record_iterator(path=filename, options=options)
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        gene_vector = example.features.feature['gene_vector'].int64_list
        print(gene_vector)


def __main__():
    tf_records_dir='/plink_tensorflow/plink_tensorflow/test/data/'

    test_dataset = MetaAnalysisDataset(raw_data_dir='/plink_tensorflow/plink_tensorflow/test/data/',
        tf_records_dir=tf_records_dir)
    for filename in test_dataset.study_records:
        decode_plink_tf_record(tf_records_dir + filename + '.tfrecords')


if __name__ == '__main__':
    __main__()
