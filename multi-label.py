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


    L_C = - (np.diag(C.sum(axis=0)) - C)
    Ec = - (F.dot(L_C.dot(F.T))).trace()

    E = El + (mu * Es) + (eta * Ec)
    return E


def propagate(l, Y, W, C, mu, eps, eta):
    S = np.diag(l)
    rL = (np.diag(W.sum(axis=0)) - W) + eps * np.eye(W.shape[0])

    n, k = Y.shape[0], Y.shape[1]

    L_C = - (np.diag(C.sum(axis=0)) - C)
    A = np.kron((S + mu * rL), np.eye(k)) - eta * np.kron(np.eye(n), L_C)

    b = (S.dot(Y)).flatten()

    F_star = np.linalg.solve(A, b)

    return F_star.reshape(Y.shape)


def main(argv):
    np.random.seed(0)

    for _ in range(2 ** 18):
        n, k = 4, 2
        mu, eps, eta = np.random.rand(), np.random.rand(), np.random.rand()

        random_matrix_W = np.random.rand(n, n)
        random_matrix_C = np.random.rand(k, k)

        W = random_matrix_W + random_matrix_W.T
        C = random_matrix_C + random_matrix_C.T

        Y = np.random.randint(2, size=(n, k))
        l = np.random.randint(2, size=n)

        F_star = propagate(l, Y, W, C, mu, eps, eta)
        lowest_energy_value = energy(F_star, l, Y, W, C, mu, eps, eta)

        noise = np.random.normal(loc=.0, scale=1e-2, size=(n, k))
        corrupted_F_star = F_star + noise

        energy_value = energy(corrupted_F_star, l, Y, W, C, mu, eps, eta)

        logging.debug('%s < %s' % (lowest_energy_value, energy_value))

        assert(lowest_energy_value <= energy_value)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])

