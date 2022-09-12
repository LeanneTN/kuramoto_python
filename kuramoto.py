import numpy as np
from scipy.integrate import odeint

class Kuramoto:

    def __init__(self, dt, coupling, T, n_nodes, natfreqs=None):
        self.dt = dt
        self.T = T
        self.coupling = coupling

        if natfreqs is not None:
            self.nafreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            self.n_nodes = n_nodes
            self.nafreqs = np.random.normal(size=self.n_nodes)

    def init_angles(self):
        '''
        randomly initialize angles (position, theta)
        '''
        return 2 * np.pi * np.random.random(size=self.n_nodes)

    def derivative(self, angle_vectors, t, adj_mat, coupling):
        '''

        :param angle_vectors:
        :param t:
        :param adj_mat:
        :param coupling:
        :return:
        '''

        assert len(angle_vectors) == len(self.nafreqs) == len(adj_mat), \
            'input dimensions do not match, check lengths'

        # 生成网格采样点矩阵
        angle_i, angle_j = np.meshgrid(angle_vectors, angle_vectors)
        # Aij * sin(j-i)
        interactions = adj_mat * np.sin(angle_j - angle_i)

        dx_dt = self.nafreqs + coupling * interactions.sum(axis=0)
        return dx_dt


