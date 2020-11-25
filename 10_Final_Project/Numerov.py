import numpy as np
import matplotlib.pyplot as plt
import Potential
import scipy.special

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


def radial_schrodinger_equation(E, l):
	"""
	Solving the radial schrodinger equation given certain E and ls(l-list)
	:param Es: Energy
	:param l: Dimensionless
	:return:
	"""

	r_min = 0.5
	r_max = 5
	r_steps = 10000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)

	# fig = plt.figure(figsize=(5, 5))  # plot the calculated values
	# fig.add_subplot(1, 1, 1)

	eff_potential = Potential.effective_potential(rs, l)
	psi = numerov(6.12 * (E - eff_potential), approx_init(r_min), approx_init(r_min + dr), dr)
	r1, r2 = r_max - dr, r_max
	u1, u2 = psi[-2], psi[-1]
	K = r1 * u2 / (r2 * u1)
	k = np.sqrt(6.12 * E)
	j1, j2 = scipy.special.spherical_jn(l, k * r1), scipy.special.spherical_jn(l, k * r2)
	n1, n2 = scipy.special.spherical_yn(l, k * r1), scipy.special.spherical_yn(l, k * r2)
	delta_l = np.arctan(((K * j1 - j2) / (K * n1 - n2)).real)

	return delta_l


def test():

	Es = np.linspace(0.1, 3.5, 200)
	delta_l = np.zeros_like(Es)
	l = 4

	for i in range(np.size(Es)):
		k = np.sqrt(6.12 * Es[i])
		delta_l[i] = 4.0 * np.pi * (2 * l + 1) * (np.sin(radial_schrodinger_equation(Es[i], l)))**2 / k / k

	fig = plt.figure(figsize=(5, 5))  # plot the calculated values
	fig.add_subplot(1, 1, 1)
	plt.plot(Es, delta_l, label=r" l={0}".format(l))
	# plt.title(r'Effective potential')
	# plt.ylim(-6, 6)
	# plt.xlabel(r"$r[\rho]$", fontsize=12)
	# plt.ylabel(r"$V_{eff}(r)$[meV]", fontsize=12)
	plt.legend()

	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.show()
	# fig.savefig("EffectivePotential.pdf", format="pdf")


test()