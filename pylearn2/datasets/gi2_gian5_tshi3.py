"""
.. todo::

    WRITEME
"""
import json
import numpy
from numpy.core.fromnumeric import choose
import itertools
import math
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy as N
import warnings
np = N
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.rng import make_np_rng


class gi2_gian5_tshi3(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : WRITEME
    center : WRITEME
    shuffle : WRITEME
    one_hot : WRITEME
    binarize : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    preprocessor : WRITEME
    fit_preprocessor : WRITEME
    fit_test_preprocessor : WRITEME
    """

    def __init__(self, file_name,
                 one_hot=None, binarize=False, start=None,
                 stop=None, axes=['b', 0, 1, 'c'],
                 preprocessor=None,
                 fit_preprocessor=False,
                 fit_test_preprocessor=False):
        self.args = locals()

        def dimshuffle(b01c):
            """
            .. todo::

                WRITEME
            """
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        
        # Path substitution done here in order to make the lower-level
        # mnist_ubyte.py as stand-alone as possible (for reuse in, e.g.,
        # the Deep Learning Tutorials, or in another package).

        topo_view,y = self.read_file(file_name)
        
        max_labels = 2
        
        one_hot_y = N.zeros((y.shape[0], max_labels), dtype='uint8')
        for i in xrange(y.shape[0]):
            one_hot_y[i, y[i]] = 1
        y = one_hot_y
        
        if one_hot is not None:
            warnings.warn("the `one_hot` parameter is deprecated. To get "
                          "one-hot encoded targets, request that they "
                          "live in `VectorSpace` through the `data_specs` "
                          "parameter of MNIST's iterator method. "
                          "`one_hot` will be removed on or after "
                          "September 20, 2014.", stacklevel=2)

        print(topo_view)
        print(y)
        m, r = topo_view.shape
#         assert c == 28
#         topo_view = topo_view.reshape(m, r, c, 1)

        super(gi2_gian5_tshi3, self).__init__(X=topo_view, y=y,
                                    axes=axes, y_labels=max_labels)

        assert not N.any(N.isnan(self.X))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)
    def read_file(self,file_name):
        with open(file_name) as file_object:
            X,y=json.load(file_object)
        return numpy.array(X,dtype='float32'),\
        	numpy.array(y,dtype='uint8'),

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        return N.clip(X * 2. - 1., -1., 1.)

    def adjust_to_be_viewed_with(self, X, other, per_example=False):
        """
        .. todo::

            WRITEME
        """
        return self.adjust_for_viewer(X)

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        args['start'] = None
        args['stop'] = None
        args['fit_preprocessor'] = args['fit_test_preprocessor']
        args['fit_test_preprocessor'] = None
        return gi2_gian5_tshi3(**args)
