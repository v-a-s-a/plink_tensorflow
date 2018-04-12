#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


def correlation_plot(test_data, gen_data):
    '''
    Plot LD matrices for test and generated data for each epoch.
    '''
    test = preprocessing.scale(test_data)
    gen = preprocessing.scale(gen_data)

    test_corr = pd.DataFrame(test).corr()
    gen_corr = pd.DataFrame(gen).corr()

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    cax1 = ax1.matshow(test_corr, vmin=-1, vmax=1)
    cax2 = ax2.matshow(gen_corr, vmin=-1, vmax=1)
    fig.colorbar(cax1)

    plt.show()
