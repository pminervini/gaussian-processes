#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging

import sys


def energy(F, l, Y, W, C, mu, eps, eta):
    S = np.diag(l)

    El = ((F - Y).T.dot(S.dot(F - Y))).trace()

    rL = (np.diag(W.sum(axis=0)) - W) + eps * np.eye(W.shape[0])
    Es = (F.T.dot(rL.dot(F))).trace()

    E = El + mu * Es

    return E


def propagate(l, Y, W, C, mu, eps, eta):
    S = np.diag(l)
    rL = (np.diag(W.sum(axis=0)) - W) + eps * np.eye(W.shape[0])

    _A = (S + mu * rL)

    k = Y.shape[1]
    A = np.kron(_A, np.eye(k))

    b = (S.dot(Y)).flatten()
    F_star = np.linalg.solve(A, b)

    return F_star.reshape(Y.shape)


def main(argv):
    np.random.seed(0)

    for _ in range(2 ** 18):

        n, k = 32, 16
        mu, eps = np.random.rand(), np.random.rand()

        random_matrix = np.random.rand(n, n)
        W = random_matrix + random_matrix.T

        Y = np.random.randint(2, size=(n, k))
        l = np.random.randint(2, size=n)

        F_star = propagate(l, Y, W, None, mu, eps, None)
        lowest_energy_value = energy(F_star, l, Y, W, None, mu, eps, None)

        noise = np.random.normal(loc=.0, scale=1e-2, size=(n, k))
        corrupted_F_star = F_star + noise

        energy_value = energy(corrupted_F_star, l, Y, W, None, mu, eps, None)
        logging.debug('%s < %s' % (lowest_energy_value, energy_value))

        assert(lowest_energy_value < energy_value)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])






