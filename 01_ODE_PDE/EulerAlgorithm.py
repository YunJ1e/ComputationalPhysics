"""
PHYSICS-514 Computational Physics
Updated: 2020/09/11
Author: Yunjie Wang
"""
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

"""
Euler Algorithm
"""


def save_data(file_name, my_data, mode='wb'):
	K = open(file_name, mode)
	np.save(K, my_data)
	K.close()


def load_data(file_name, number_of_objs=1, mode='rb'):
	H = open(file_name, mode)
	J = np.load(H)
	H.close()
	return J


def plot_p1(k=1):
	t = load_data("p1_time.npy")
	y_exact = load_data("p1_y_exact.npy")
	y = load_data("p1_y.npy")
	y_backward = load_data("p1_y_backward.npy")
	y_RK4 = load_data("p1_y_RK4.npy")
	y_lf = load_data("p1_y_lf.npy")

	plt.plot(t[::2], y[::2], 'b*', markersize=4.0, label="Numerical Solution 01")
	plt.plot(t[::2], y_backward[::2], 'r*', markersize=4.0, label="Numerical Solution 02")
	plt.plot(t[::2], y_RK4[::2], 'g*', markersize=4.0, label="RK4")
	plt.plot(t[::2], y_lf[::2], 'y*', markersize=4.0, label="Leapfrog")
	plt.plot(t, y_exact, 'k-', markersize=1.0, label="Exact Solution")
	plt.title(r'$\frac{{d^{{2}}y}}{{dt^{{2}}}} = -ky$, k = {0}'.format(k))
	plt.legend(loc="upper left")
	plt.xlabel("Time")
	plt.ylabel("Distance")
	#plt.show()
	plt.savefig("Problem_01.pdf", format="pdf")
	plt.close()
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
	# f = open('test.npy', 'wb')
	# np.save(f, t)
	save_data("p1_time.npy", t)
	v = np.zeros(n)
	y = np.zeros(n)
	y_backward = np.zeros(n)
	v_backward = np.zeros(n)
	v_RK4 = np.zeros(n)
	y_RK4 = np.zeros(n)
	v_lf = np.zeros(n)
	y_lf = np.zeros(n)

	y_exact = (v0/omega) * np.sin(omega * t) + y0 * np.cos(omega * t)
	save_data("p1_y_exact.npy", y_exact)
	# f = open('test1.npy', 'wb')
	# np.save(f, y_exact)
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

	save_data("p1_y.npy", y)
	save_data("p1_y_backward.npy", y_backward)
	save_data("p1_y_RK4.npy", y_RK4)
	save_data("p1_y_lf.npy", y_lf)
	# plt.plot(t[::2], y[::2], 'b*', markersize=4.0, label="Numerical Solution 01")
	# plt.plot(t[::2], y_backward[::2], 'r*', markersize=4.0, label="Numerical Solution 02")
	# plt.plot(t[::2], y_RK4[::2], 'g*', markersize=4.0, label="RK4")
	# plt.plot(t[::2], y_lf[::2], 'y*', markersize=4.0, label="Leapfrog")
	# plt.plot(t, y_exact, 'k-', markersize=1.0, label="Exact Solution")
	# plt.title(r'$\frac{{d^{{2}}y}}{{dt^{{2}}}} = -ky$, k = {0}'.format(k))
	# plt.legend(loc="upper left")
	# plt.xlabel("Time")
	# plt.ylabel("Distance")
	# plt.show()


"""
Test Run for Problem 1
"""
euler_method(k=25)
plot_p1(k=25)


# y'' = -g(constant)
def func(alpha=69.0, drawing=False):
	alpha = math.radians(alpha)
	distance = 1500
	n = 10001
	t0 = 0
	v0 = 150
	y0 = 0
	tf = 2 * v0 / 10
	deltat = (int(tf) + 1 - t0) / (n - 1)
	# print(deltat)
	t = np.linspace(t0, tf, n)
	v = np.zeros(n)
	y = np.zeros(n)
	x = np.zeros(n)
	# print(t)
	y[0] = y0
	v[0] = v0 * np.sin(alpha)
	# print(v[0])
	x[0] = 0
	for i in range(1, n):
		# Forward Euler
		v[i] = v[i - 1] + deltat * (-10)
		y[i] = y[i - 1] + deltat * v[i - 1]
		# print(y[i])
		x[i] = x[i - 1] + deltat * v0 * np.cos(alpha)
		if y[i] < 0:
			if not drawing:
				return x[i]
			else:
				break
	# y_exact = (3 - np.sqrt(5)) * (1500 - x) * x / (6 * 25 * 2 * 10)
	save_data("p2_x.npy", x)
	save_data("p2_y.npy", y)


def secant():
	x0 = 69.0
	x1 = 70.0
	f_x0 = func(x0) - 1500.0
	f_x1 = func(x1) - 1500.0
	# print(f_x0, f_x1)
	iteration_counter = 0

	while abs(f_x1) > 10**(-10) and iteration_counter < 100:
		denominator = float((f_x1 - f_x0) / (x1 - x0))
		x = float(x1 - (f_x1 / denominator))

		x0 = x1
		x1 = x
		f_x0 = f_x1
		f_x1 = func(x1) - 1500.0
		iteration_counter += 1

	func(alpha=x, drawing=True)


def plot_p2():
	x = load_data("p2_x.npy")
	y = load_data("p2_y.npy")
	plt.plot(x[::100], y[::100], 'b*', label="Numerical Solution of BVP")
	plt.title("The Shooting Problem")
	plt.legend(loc="upper left")
	plt.xlabel("X(Distance)")
	plt.ylabel("Y(Height)")
	# plt.show()
	plt.savefig("Problem_02.pdf", format="pdf")
	plt.close()

"""
Test Run for Problem 2
"""
secant()
plot_p2()

def func1():
	plt.ion()
	Nx = 200
	Nt = 20000
	T = 1
	a = 0.1
	x = np.linspace(-1, 1, Nx + 1)
	dx = x[1] - x[0]
	t = np.linspace(0, T, Nt + 1)
	dt = t[1] - t[0]
	F = a * dt / dx ** 2
	u = np.zeros(Nx + 1)
	u_1 = np.zeros(Nx + 1)

	for i in range(90, 110 + 1):
		u_1[i] = 1
	save_data("p3_u.npy", u_1)
	plt.plot(x, u, 'b*', markersize=4.0, label="Numerical Solution")
	plt.savefig("Problem_03_Initial.pdf", format="pdf")
	plt.close()
	for n in range(0, Nt):

		for i in range(1, Nx):
			u[i] = u_1[i] + F * (u_1[i - 1] - 2 * u_1[i] + u_1[i + 1])

		# Insert boundary conditions
		u[0] = 0
		u[-1] = 0

		# Update u_1 before next step
		u_1[:] = u
		save_data("p3_u.npy", u_1)
		if n % 1000 == 0:
			plt.title("{0}".format(n))
			plt.ylim([-1, 1])
			plt.plot(x, u, 'b*', markersize=4.0, label="Numerical Solution 01")
			plt.draw()
			plt.pause(1)
			plt.savefig("Problem_03_{0}.pdf".format(n), format="pdf")
			plt.clf()
	plt.plot(x, u, 'b*', markersize=4.0, label="Numerical Solution")
	plt.savefig("Problem_03_Final.pdf", format="pdf")
	plt.close()

# def func3(Nt = 20000):
# 	data_list = []
#
# 	for i in range(Nt+1):
# 		data_list.append(load_data("p3_u.npy"))
# 	data_list = np.array(data_list)
# 	# print(np.array(data_list))
# 	return data_list

"""
Test Run for Problem 3
"""
func1()
# func3()
