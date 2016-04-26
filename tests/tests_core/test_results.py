# fileencoding: utf8
import os
import unittest
import warnings

import numpy as np
import pandas as pd

import expan.core.results as r
from expan.core.experiment import Experiment
from tests.tests_core.test_data import generate_random_data

reload(r)

data_dir = os.getcwd() + '/tests/tests_core/'  # TODO: adjust depending on where we're called from?


def generate_random_results():
    # TODO
    pass


def load_example_results():
    """
    This just loads example data that was created with the
    generate_results_from_retargeting function, so that we need not load
    the full data from that experiment in order to test.

    Returns Results object.
    """

    example_fname = 'example_results.h5'
    example_fpath = os.path.join(data_dir, example_fname)
    return r.from_hdf(example_fpath)


def generate_results_from_retargeting(df):
    """
    Input: ckpis from Retargeting experiment.

    Just experimenting with the data structure.
    This should give out a format that should be appropriate for:
     - delta kpi
     - sga
     - time profile
    """
    metadata = {
        'experiment': 'Retargeting NL',
        'load_time': pd.Timestamp.now('UTC'),
        'source_type': 'file_local',
        'source': 'exasol',
        'cached': True
    }
    # res = df.reset_index().groupby(['treatment','start_segment'])[[
    #      'orders','net_sales','pcii']].agg(['mean','sum','var','count'])
    res = df.reset_index().pivot_table(
        values=['orders', 'net_sales', 'pcii'],  # just a convenience, subset
        index=['treatment', 'start_segment'],
        # columns=['treatment'],
        aggfunc=[np.mean, sum, np.var])

    res = res.stack('metric')

    res.columns.set_names('statistic', inplace=True)

    res['subgroup_metric'] = 'start_segment'
    res.index.rename('subgroup', level='start_segment', inplace=True)
    res.set_index('subgroup_metric', append=True, inplace=True)

    res.index.rename('variant', 'treatment', inplace=True)

    res = res.reorder_levels(
        ['metric', 'subgroup_metric', 'subgroup', 'variant'])

    res = pd.DataFrame(res.stack(level='statistic'), columns=['value'])
    res['pctile'] = np.nan
    # res['unit']='€'; res.loc['orders','unit']='orders'

    res = res.unstack(level='variant')
    res = res.swaplevel(0, 1, axis=1)

    mm = df.groupby(level='treatment')['orders', 'net_sales', 'pcii'].agg(
        ['mean', 'sum', 'var'])
    mm.index.name = 'variant'
    mm.columns.names = ['metric', 'statistic']
    mm = mm.T

    nn = pd.DataFrame(mm.stack(), columns=['value'])
    nn = nn.unstack('variant')
    nn = nn.swaplevel(0, 1, 1)
    hh = nn.stack('variant')

    hh['subgroup_metric'] = '-'
    hh['subgroup'] = np.nan
    hh['pctile'] = np.nan

    jj = hh.reset_index().set_index(
        ['metric', 'subgroup_metric', 'subgroup', 'statistic', 'pctile', 'variant']
    ).unstack('variant').swaplevel(0, 1, 1)

    final = pd.concat([res, jj])

    final.sortlevel(axis=0, inplace=True, sort_remaining=True)
    final.sortlevel(axis=1, inplace=True, sort_remaining=True)

    return r.Results(final, metadata)


class ResultsTestCase(unittest.TestCase):
    """
    Defines the setUp() and tearDown() functions for the results test cases.
    """

    def setUp(self):
        """
        Load the needed datasets for all TestCases and set the random
        seed so that randomized algorithms show deterministic behaviour.
        """
        np.random.seed(0)
        self.data = Experiment('B', *generate_random_data())
        # Create time column. TODO: Do this nicer
        self.data.kpis['time_since_treatment'] = \
            self.data.features['treatment_start_time']
        # Make time part of index
        self.data.kpis.set_index('time_since_treatment', append=True, inplace=True)

    def tearDown(self):
        """
        Clean up after the test
        """
        # TODO: find out if we have to remove data manually
        pass


class ResultsClassTestCase(ResultsTestCase):
    def testExampleResults(self):
        h5py_available = False
        import imp
        try:
            imp.find_module('h5py')
            imp.find_module('tables')
            h5py_available = True
        except Exception:
            warnings.warn(
                """Could not import h5py or tables module. Skipping
                testExampleResults(). Please make sure that you have the h5py
                and tables packages installed."""
            )

        if h5py_available:
            aa = load_example_results()

            # check mean
            df = aa.statistic('delta', 'mean', 'orders')
            np.testing.assert_almost_equal(df.iloc[0],
                                           np.array([0.115315, 0.118150, 0.116237, 0.117082]), decimal=5)
            # check var
            df = aa.statistic('delta', 'var', 'net_sales')
            np.testing.assert_almost_equal(df.iloc[0],
                                           np.array([5775.809079, 5519.640915, 5712.736931, 5941.701489]), decimal=5)

            self.assertEquals(aa.metadata['source'], 'dwh_copy')
            self.assertEquals(aa.metadata['source_type'], 'exasol')
            self.assertEquals(aa.metadata['some_time'].year, 2015)

        # self.assertEqual(aa.sample_size_baseline, 3892)
        # self.assertEqual(aa.sample_size[1], 6108)

    def test_relative_uplift_delta(self):
        """Check if the calculation of relative uplift for delta results is
        correct.
        """
        res = self.data.delta()
        df = res.relative_uplift('delta', 'normal_same')
        np.testing.assert_almost_equal(df, np.array([[-4.219601, 0]]), decimal=5)


if __name__ == '__main__':
    unittest.main()
