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
        theta is initial position of one oscillator
        '''
        return 2 * np.pi * np.random.random(size=self.n_nodes)

    def derivative(self, angle_vectors, t, adj_mat, coupling):
        '''
        derivative function. Original equation can watch the pictures
        under image directory in same name
        :param angle_vectors:
        :param t:
        :param adj_mat: A_ij in original equation, means adjacent matrix
        :param coupling: k/M in original equation
        :return: matrix
        '''

        assert len(angle_vectors) == len(self.nafreqs) == len(adj_mat), \
            'input dimensions do not match, check lengths'

        # 生成网格采样点矩阵
        # i为横向拼接，j为竖向拼接
        angle_i, angle_j = np.meshgrid(angle_vectors, angle_vectors)
        # Aij * sin(j-i)
        interactions = adj_mat * np.sin(angle_j - angle_i)

        # omega + K/M * Aij * sin(j - i)
        dx_dt = self.nafreqs + coupling * interactions.sum(axis=0)
        return dx_dt

    def integrate(self, angle_vectors, adj_matrix):
        '''
        updates all states by integrating all nodes' states in matrix
        :param angle_vectors: vectors of each angle
        :param adj_matrix: adjacent matrix
        :return:
        '''
        # number of incoming interactions
        n_interactions = (adj_matrix != 0).sum(axis=0)
        # normalize coupling by number of interactions
        coupling = self.coupling / n_interactions

        t = np.linspace(0, self.T, int(self.T/self.dt))
        # derivative是微分方程, angle_vectors是微分方程的初值, t是微分的自变量
        # odeint是一个用于求解微分方程的函数
        time_series = odeint(self.derivative, angle_vectors, t, args=(adj_matrix, coupling))
        return time_series.T

    def run(self, adj_matrix=None, angle_vectors=None):
        '''

        :param adj_matrix: adjacent matrix, nd array
        2D: show the connectivity of each node
        1D: states vectors of nodes shows the position in radians
        :param angle_vectors: angle vectors
        :return: integrated matrix, 2D nd array
        '''
        if angle_vectors is None:
            angle_vectors = self.init_angles()

        return self.integrate(angle_vectors, adj_matrix)

