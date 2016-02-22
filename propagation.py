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


def likelihood(f, l, R, mu, eps, sigma2, lambda_1=1e-4):
    # The similarity matrix W is a linear combination of the slices in R
    W = T.tensordot(R, mu, axes=1)

    # The following indices correspond to labeled and unlabeled examples
    labeled = T.eq(l, 1).nonzero()

    # Calculating the graph Laplacian of W
    D = T.diag(W.sum(axis=0))
    L = D - W

    # The Covariance (or Kernel) matrix is the inverse of the (regularized) Laplacian
    epsI = eps * T.eye(L.shape[0])
    rL = L + epsI
    Sigma = nlinalg.matrix_inverse(rL)

    # The marginal density of labeled examples uses Sigma_LL as covariance (sub-)matrix
    Sigma_LL = Sigma[labeled][:, labeled][:, 0, :]

    # We also consider additive Gaussian noise with variance sigma2
    K_L = Sigma_LL + (sigma2 * T.eye(Sigma_LL.shape[0]))

    # Calculating the inverse and the determinant of K_L
    iK_L = nlinalg.matrix_inverse(K_L)
    dK_L = nlinalg.det(K_L)

    f_L = f[labeled]

    # The (L1-regularized) log-likelihood is given by the summation of the following four terms
    term_A = - (1 / 2) * f_L.dot(iK_L.dot(f_L))
    term_B = - (1 / 2) * T.log(dK_L)
    term_C = - (1 / 2) * T.log(2 * np.pi)
    term_D = - lambda_1 * T.sum(abs(mu))

    return term_A + term_B + term_C + term_D


def propagate(f, l, R, mu, eps):
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

    # Computing the inverse of the (regularized) Laplacian iA = (L_UU + epsI)^-1
    epsI = eps * T.eye(L_UU.shape[0])
    rL_UU = L_UU + epsI
    iA = nlinalg.matrix_inverse(rL_UU)

    # Computing W_UL (the similarity matrix between unlabeled and labeled examples)
    W_UL = W[unlabeled][:, labeled][:, 0, :]
    f_L = f[labeled]

    # f* = (L_UU + epsI)^-1 W_UL f_L
    f_star = iA.dot(W_UL.dot(f_L))

    return f_star


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    # Training labels, similarity matrix and weight of the regularization term
    f, R, mu, eps = T.dvector('f'), T.dtensor3('R'), T.dvector('mu'), T.dscalar('eps')
    sigma2 = T.dscalar('sigma2')

    # Indices of labeled examples
    l = T.ivector('l')

    f_star = propagate(f, l, R, mu, eps)
    ll = likelihood(f, l, R, mu, eps, sigma2)

    propagate_f = theano.function([f, l, R, mu, eps], f_star, on_unused_input='warn')
    likelihood_function = theano.function([f, l, R, mu, eps, sigma2], ll, on_unused_input='warn')

    ll_grad = T.grad(ll, [mu, eps, sigma2])
    likelihood_gradient_function = theano.function([f, l, R, mu, eps, sigma2], ll_grad, on_unused_input='warn')

    nb_nodes = 64

    R = np.zeros((nb_nodes, nb_nodes, 1))
    even_edges = [(i, i + 2) for i in range(0, nb_nodes, 2) if (i + 2) < nb_nodes]
    odd_edges = [(i, i + 2) for i in range(1, nb_nodes, 2) if (i + 2) < nb_nodes]

    for source, target in even_edges + odd_edges:
        R[source, target, 0], R[target, source, 0] = 1.0, 1.0

    mu = np.ones(1)
    eps = 1e-2
    sigma2 = 1e-6

    f = np.array([+ 1.0, - 1.0] + ([.0] * (nb_nodes - 2)))
    l = np.array(f != 0, dtype='int8')

    print(propagate_f(f, l, R, mu, eps))

    learning_rate = 1e-2

    for i in range(1024):
        ll_value = likelihood_function(f, l, R, mu, eps, sigma2)
        print('LL [%d]: %s' % (i, ll_value))

        grad_value = likelihood_gradient_function(f, l, R, mu, eps, sigma2)

        mu += learning_rate * grad_value[0]
        eps += max(1e-6, learning_rate * grad_value[1])
        sigma2 += max(1e-6, learning_rate * grad_value[2])

    print('Mu: %s' % str(mu))
    print('Eps: %s' % str(eps))
    print('Sigma^2: %s' % str(sigma2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
