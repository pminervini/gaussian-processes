#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T

import theano.tensor.nlinalg as nlinalg

import sys
import argparse
import logging

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


def propagation(f, l, R, mu, eps):
    # The similarity matrix W is a linear combination of the slices in R
    W = T.tensordot(R, mu, axes=1)

    # The following indices correspond to labeled and unlabeled examples
    labeled = T.eq(l, 1).nonzero()
    unlabeled = T.eq(l, 0).nonzero()

    # Calculating the graph Laplacian of W
    D = T.diag(W.sum(axis=0))
    L = D - W

    # Computing L_UU (the Laplacian over unlabeled examples)
    L_UU = L[unlabeled][:, unlabeled][:, 0, :]

    # Computing iA = (L_UU + epsI)^-1
    epsI = eps * T.eye(L_UU.shape[0])
    iA = nlinalg.matrix_inverse(L_UU + epsI)

    # Computing W_UL (the similarity matrix between unlabeled and labeled examples)
    W_UL = W[unlabeled][:, labeled][:, 0, :]
    f_L = f[labeled]

    f_star = iA.dot(W_UL.dot(f_L))

    return f_star


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    # Training labels, similarity matrix and weight of the regularization term
    f, R, mu, eps = T.dvector('f'), T.dtensor3('R'), T.dvector('mu'), T.dscalar('eps')
    # Indices of labeled examples
    l = T.ivector('l')

    f_star = propagation(f, l, R, mu, eps)
    propagation_function = theano.function([f, l, R, mu, eps], f_star, on_unused_input='warn')

    nb_nodes = 64

    R = np.zeros((nb_nodes, nb_nodes, 1))
    even_edges = [(i, i + 2) for i in range(0, nb_nodes, 2) if (i + 2) < nb_nodes]
    odd_edges = [(i, i + 2) for i in range(1, nb_nodes, 2) if (i + 2) < nb_nodes]

    for source, target in even_edges + odd_edges:
        R[source, target, 0], R[target, source, 0] = 1.0, 1.0

    mu = np.ones(1)

    f = np.array([+ 1.0, -1.0] + ([.0] * (nb_nodes - 2)))
    l = np.array(f != 0, dtype='int8')

    print(propagation_function(f, l, R, mu, 1e-2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
