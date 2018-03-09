#!/usr/bin/env python
'''
Creating a feed for PLINK formatted data in a meta-analysis setting.

Some things we would like to be able to do:
  * Split into test/train studies.
  * For each study, yield a batch of samples.
  * Optionally subset to variants in a given region.
'''
import os
import tensorflow as tf

from pandas_plink import read_plink
from sklearn.model_selection import train_test_split
from random import shuffle
import numpy as np
import dask.array as da


def minibatch(X, batch_size, shuffle=True):
    '''
    Yield minibatches of a numpy array for training.
    '''
    n = X.shape[0]
    idxs = np.arange(n)
    if shuffle:
        np.random.shuffle(idxs)

    for i in iter(range((n // batch_size))):
        batch_start_idx = idxs[i * batch_size]
        data = X[batch_start_idx:(batch_start_idx + batch_size), :]
        yield data


class MetaAnalysisDataset:

    def __init__(self, test_prop=0.8, raw_data_dir='/plink_tensorflow/data/'):
        '''
        Map a directory of plink files to dask arrays and pandas dataframes.

        @test_prop: The rough proportion of sample to dedicate to training.
        @raw_data_dir: Directory containing PLINK formatted files for each study.
        '''

        self.test_prop = test_prop

        # map the input files into pandas dataframes and dask arrays
        root, dirs, files = next(os.walk(raw_data_dir))
        study_plink_prefixes = [root+f.replace('.bim', '') for f in files if f.endswith('.bim')]
        
        # read_plink -> (bim, fam, G)
        print('Generating Dask arrays from study PLINK files...')
        self.study_arrays = {os.path.basename(f): read_plink(f) for f in study_plink_prefixes}
        print('Done')

        # split into test/training sets
        self.test_train_split()

        ## TODO: check that all studies contain the same variants


    def test_train_split(self):
        '''
        We split test/train by study.
        We want to make sure that proportion of test/train is related to number of
        samples in each study.

        Greedy solution: Randomly select datasets for testing until the proportion
            of the test set is exceeded.
        '''
        total_sample_size = float(sum([x[1].shape[0] for x in self.study_arrays.values()]))
        self.test_studies = {}
        self.train_studies = {}
        train_set_size = 0.0
        test_set_size = 0.0
        studies = self.study_arrays.keys()
        shuffle(studies)
        for study in studies:
            if train_set_size < (total_sample_size * self.test_prop):
                self.train_studies[study] = self.study_arrays[study]
                train_set_size += self.study_arrays[study][1].shape[0]
                self.m_variants = self.study_arrays[study][0].shape[0]
            else:
                self.test_studies[study] = self.study_arrays[study]
        
        # some logging
        print('Training set:\t{} studies and {:.3f} of samples in training set.'.format(len(self.train_studies.keys()), train_set_size/total_sample_size))
        print('Testing set: \t{} studies, {:.3f} of samples in test set.'.format(len(self.test_studies.keys()), test_set_size/total_sample_size))


    def test_set(self):
        gene_matrix = da.concatenate([G for (bim, fam, G) in self.test_studies.values()], axis=0)
        gene_matrix = da.transpose(gene_matrix).to_dask_dataframe()
        return gene_matrix.fillna(gene_matrix.mean(axis=0), axis=0)


    def train_set_minibatches(self, batch_size=10):
        '''
        Yield batches of samples from the studies in the test and train datasets.
        '''
        for study, (bim, fam, G) in self.train_studies.items():
            for batch in minibatch(da.transpose(G), batch_size=batch_size):
                gene_matrix = batch.to_dask_dataframe()
                yield gene_matrix.fillna(gene_matrix.mean(axis=0), axis=0)
