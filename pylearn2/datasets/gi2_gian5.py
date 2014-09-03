"""
.. todo::

    WRITEME
"""
import json
import numpy
from numpy.core.fromnumeric import choose
import itertools
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


class gi2_gian5(dense_design_matrix.DenseDesignMatrix):
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

    def __init__(self, which_set, shuffle=False,
                 one_hot=None, binarize=False, start=None,
                 stop=None, axes=['b', 0, 1, 'c'],
                 choose=100,
                 preprocessor=None,
                 fit_preprocessor=False,
                 fit_test_preprocessor=False):
        self.args = locals()

        if which_set not in ['train', 'test']:
            if which_set == 'valid':
                raise ValueError(
                    "There is no such thing as the MNIST validation set. MNIST"
                    "consists of 60,000 train examples and 10,000 test"
                    "examples. If you wish to use a validation set you should"
                    "divide the train set yourself. The pylearn2 dataset"
                    "implements and will only ever implement the standard"
                    "train / test split used in the literature.")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        def dimshuffle(b01c):
            """
            .. todo::

                WRITEME
            """
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        path = "/home/Ihc/git/huan1-ik8_gian5-kiu3/mt/"
        if which_set == 'train':
            im_path = path + 'h.q'
            label_path = path + 'h.a'
        else:
            assert which_set == 'test'
            im_path = path + 't.q'
            label_path = path + 't.a'
        # Path substitution done here in order to make the lower-level
        # mnist_ubyte.py as stand-alone as possible (for reuse in, e.g.,
        # the Deep Learning Tutorials, or in another package).

        topo_view = self.read_json(im_path,dtype='float32',n=choose)
        y = self.read_json(label_path,dtype='uint8')
        if binarize:
            topo_view = (topo_view > 0.5).astype('float32')

        max_labels = 2
        if one_hot is not None:
            warnings.warn("the `one_hot` parameter is deprecated. To get "
                          "one-hot encoded targets, request that they "
                          "live in `VectorSpace` through the `data_specs` "
                          "parameter of MNIST's iterator method. "
                          "`one_hot` will be removed on or after "
                          "September 20, 2014.", stacklevel=2)

        print(y)
        m, r = topo_view.shape
        assert r == 12+choose+choose
#         assert c == 28
#         topo_view = topo_view.reshape(m, r, c, 1)

        if shuffle:
            self.shuffle_rng = make_np_rng(None, [1, 2, 3], which_method="shuffle")
            for i in xrange(m):
                j = self.shuffle_rng.randint(m)
                # Copy ensures that memory is not aliased.
                tmp = topo_view[i, :].copy()
#                 print(tmp)
                topo_view[i, :] = topo_view[j, :]
                topo_view[j, :] = tmp
                # Note: slicing with i:i+1 works for one_hot=True/False
                tmp = y[i:i+1].copy()
                y[i] = y[j]
                y[j] = tmp
        if which_set == 'train':
            assert m == 17887
        elif which_set == 'test':
            assert m == 3741
        else:
            assert False

        super(gi2_gian5, self).__init__(X=topo_view, y=y,
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

        if which_set == 'test':
            assert fit_test_preprocessor is None or \
                (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)
    def read_json(self,path, dtype='float32',n=None):
        with open(path) as file_object:
            array=json.load(file_object)
        if n!=None:
            array=self.choose_feature(array, n)
        else:
            chosed=[]
            for q in array:
                chosed.append([q])
            array=chosed
        return numpy.array(array,dtype=dtype)
    def choose_feature(self,qq,n=0):
        chosed=[]
        for q in qq:
            q[0]=-q[0]/10.0
            q[2]=-q[2]/10.0
            for i in itertools.chain(xrange(6,10),xrange(14,14+n)):
                q[i]=q[i]/q[1]
            for i in itertools.chain(xrange(10,14),xrange(1014,1014+n)):
                q[i]=q[i]/q[3]
            q[1]/=1000.0
            q[3]/=1000.0
            chosed.append(q[0:4] + q[6:14] + q[14:14 + n] + q[1014:1014 + n])
        return chosed

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
        return gi2_gian5(**args)
