"""
PHYSICS-514 Computational Physics
Updated: 2020/09/11
Author: Yunjie Wang
"""
import numpy as np
from matplotlib import pyplot as plt
"""
Euler Algorithm
"""


# y'' = -ky
def euler_method(k=1, t0=0, v0=0, y0=1, number_of_period=10, n=1001):
	"""

	:param k:
	:param t0:
	:param v0:
	:param y0:
	:param number_of_period:
	:param n:
	:return:
	"""
	omega = np.sqrt(k)
	period = 2 * np.pi / omega
	tf = number_of_period * period
	deltat = (tf - t0) / (n - 1)

	t = np.linspace(t0, tf, n)
	v = np.zeros(n)
	y = np.zeros(n)
	y_backward = np.zeros(n)
	v_backward = np.zeros(n)
	v_RK4 = np.zeros(n)
	y_RK4 = np.zeros(n)
	v_lf = np.zeros(n)
	y_lf = np.zeros(n)

	y_exact = (v0/omega) * np.sin(omega * t) + y0 * np.cos(omega * t)
	y[0] = y0
	y_backward[0] = y0
	y_RK4[0] = y0
	y_lf[0] = y0

	v[0] = v0
	v_backward[0] = v0
	v_RK4[0] = v0
	v_lf[0] = v0 - deltat * (k * y0)/2

	for i in range(1, n):
		# Forward Euler
		y[i] = y[i - 1] + deltat * v[i - 1]
		v[i] = v[i - 1] - deltat * k * y[i - 1]

		# Backward Euler
		# y_backward[i] = y_backward[i - 1] + deltat * v_backward[i - 1]
		# v_backward[i] = v_backward[i - 1] - deltat * k * y_backward[i]

		y_bar = y_backward[i - 1] + deltat * v_backward[i - 1]
		v_backward[i] = v_backward[i - 1] - deltat * k * y_bar
		y_backward[i] = y_backward[i - 1] + deltat * v_backward[i]

		# Leapfrog
		v_lf[i] = v_lf[i - 1] - deltat * k * y_lf[i - 1]
		y_lf[i] = y_lf[i - 1] + deltat * v_lf[i]

		# Runge-Kutta 4th
		k1y = deltat * v_RK4[i - 1]
		k1z = deltat * (-k * y_RK4[i - 1])

		k2y = deltat * (v_RK4[i - 1] + k1z/2)
		k2z = deltat * (-k * (y_RK4[i - 1] + k1y/2))

		k3y = deltat * (v_RK4[i - 1] + k2z/2)
		k3z = deltat * (-k * (y_RK4[i - 1] + k2y/2))

		k4y = deltat * (v_RK4[i - 1] + k3z)
		k4z = deltat * (-k * (y_RK4[i - 1] + k3y))

		v_RK4[i] = v_RK4[i - 1] + (k1z + 2 * k2z + 2 * k3z + k4z)/6
		y_RK4[i] = y_RK4[i - 1] + (k1y + 2 * k2y + 2 * k3y + k4y)/6

	plt.plot(t[::2], y[::2], 'b*', markersize=4.0, label="Numerical Solution 01")
	plt.plot(t[::2], y_backward[::2], 'r*', markersize=4.0, label="Numerical Solution 02")
	plt.plot(t[::2], y_RK4[::2], 'g*', markersize=4.0, label="RK4")
	plt.plot(t[::2], y_lf[::2], 'y*', markersize=4.0, label="Leapfrog")
	plt.plot(t, y_exact, 'k-', markersize=1.0, label="Exact Solution")
	plt.title(r'$\frac{{d^{{2}}y}}{{dt^{{2}}}} = -ky$, k = {0}'.format(k))
	plt.legend(loc="upper left")
	plt.xlabel("Time")
	plt.ylabel("Distance")
	plt.show()


euler_method(k=25)