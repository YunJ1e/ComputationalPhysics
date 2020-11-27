import numpy as np
from Potential import effective_potential
import matplotlib.pyplot as plt
from matplotlib import animation


def approx_init(rmin):
	"""
	Calculate the first and second point according to certain approximation rules
	:param rmin: Starting point(or the point after it) of the integrate period
	:return: The approximation result
	"""
	C = np.sqrt(5.9 * 6.12 / 25.0)
	return np.exp(-C * rmin ** (-5))


def numerov_step(k0, k1, k2, psi0, psi1, dx):
	"""
	Calculate the single Numerov step
	:param k0: k(n-1) in the equation 1.7
	:param k1: k(n) in the equation 1.7
	:param k2: k(n+1) in the equation 1.7
	:param psi0: psi(n-1) in the equation 1.7
	:param psi1: psi(n) in the equation 1.7
	:param dx: step length
	:return: psi2
	"""
	dx_square = dx ** 2
	c0 = - (1 + dx_square * k0 / 12.0)
	c1 = 2.0 * (1 - 5 * dx_square * k1 / 12.0)
	c2 = 1 + dx_square * k2 / 12.0
	return (c1 * psi1 + c0 * psi0) / c2


def numerov(K, psi0, psi1, dx):
	"""
	Solving specific differential equation using Numerov algorithm
	:param K: a numpy list containing k info in psi'' + k * psi = 0
	:param psi0: The first initial point
	:param psi1: The first initial point
	:param dx: step length
	:return: psi, a wave function
	"""
	n = np.size(K)
	psi = np.zeros(n, dtype=np.complex128)
	psi[0] = psi0
	psi[1] = psi1
	for i in range(2, n):
		psi[i] = numerov_step(K[i - 2], K[i - 1], K[i], psi[i - 2], psi[i - 1], dx)
	return psi


def radial_schrodinger_equation(rs, dr, E, l):
	"""
	Solving the radial schrodinger equation given certain E and ls(l-list)
	:param rs: A list of position is in unit of rho
	:param dr: The step length
	:param E: Energy
	:param l: Dimensionless
	:return:
	"""
	r_min = rs[0]
	eff_potential = effective_potential(rs, l)
	psi = numerov(6.12 * (E - eff_potential), approx_init(r_min), approx_init(r_min + dr), dr)
	return psi


def animation_plot(eps=5.9):
	r_min = 0.5
	r_max = 7.5
	r_steps = 1000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)

	fig = plt.figure(figsize=(8, 8))
	fig.suptitle(r"Radial Wave Function $\chi(R) / R$  Versus R @ different E")
	axs = []
	lines = []
	my_texts = []
	for i in range(1, 10):
		ax = fig.add_subplot(3, 3, i)
		ax.fill_between(rs, effective_potential(rs, i - 1), color="silver")
		my_text = ax.text(0.22, 0.10, "", transform=ax.transAxes, bbox=dict(facecolor='green', alpha=0.5))
		line, = ax.plot([], [], lw=2)
		axs.append(ax)
		lines.append(line)
		my_texts.append(my_text)

	for i in range(9):
		axs[i].set_xlim(r_min, r_max)
		axs[i].set_ylim(-6, 6)

	def animate(i):
		for j in range(9):
			my_texts[j].set_text("l = {0},E = {1:5.2f}meV".format(j, 0.1 + 0.01 * i))
			lines[j].set_data(rs, 1e3 * np.real(radial_schrodinger_equation(rs, dr, 0.1 + 0.01 * i, j)) / np.sum(np.absolute(radial_schrodinger_equation(rs, dr, 0.1 + 0.01 * i, j))))
		return lines + my_texts

	my_animation = animation.FuncAnimation(fig, animate, frames=400, interval=2, blit=True, repeat=True)
	plt.show()


animation_plot()
