import numpy as np
from Potential import effective_potential
import matplotlib.pyplot as plt
from matplotlib import animation
from Potential import prefactor
from Particle import Particle
plt.rcParams['animation.ffmpeg_path'] = "C:\FFmpeg\\bin\\ffmpeg.exe"

def approx_init(rmin, particle_object):
	"""
	Calculate the first and second point according to certain approximation rules
	:param rmin: Starting point(or the point after it) of the integrate period
	:param particle_object: contains necessary parameters for the calculation
	:return: The approximation result
	"""
	eps = particle_object.eps
	two_m_hbar_square = prefactor(particle_object)
	C = np.sqrt(eps * two_m_hbar_square / 25.0)
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


def radial_schrodinger_equation(rs, dr, E, l, particle_object):
	"""
	Solving the radial schrodinger equation given certain E and ls(l-list)
	:param rs: A list of position is in unit of rho
	:param dr: The step length
	:param E: Energy
	:param l: Dimensionless
	:param particle_object: contains necessary parameters for the calculation
	:return:
	"""
	eps = particle_object.eps
	two_m_hbar_square = prefactor(particle_object)
	r_min = rs[0]
	eff_potential = effective_potential(rs, l, particle_object)
	psi = numerov(two_m_hbar_square * (E - eff_potential), approx_init(r_min, particle_object), approx_init(r_min + dr, particle_object), dr)
	return psi


def animation_plot(particle_object):
	"""
	Generate the animation of the radial wave function for certain particle object
	:param particle_object: the data structure containing necessary information of the scattering particle
	:return: None
	"""
	Writer = animation.writers['ffmpeg']
	writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
	eps = particle_object.eps
	r_min = 0.5
	r_max = 7.5
	r_steps = 1000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)

	fig = plt.figure(figsize=(8, 8))
	fig.suptitle(r"Radial Wave Function $\chi(R) / R$  Versus R @ different E")
	axs = []
	lines = []
	energys = []
	my_texts = []
	shades = []
	for i in range(1, 10):
		ax = fig.add_subplot(3, 3, i)
		# potential_shade = ax.fill_between(rs, effective_potential(rs, i - 1), color="silver")
		# potential_shade = ax.fill_between([], [], color="silver")
		my_text = ax.text(0.22, 0.10, "", transform=ax.transAxes, bbox=dict(facecolor='green', alpha=0.5))
		potential_shade, = ax.plot([], [], lw=1)
		line, = ax.plot([], [], lw=2)
		energy, = ax.plot([], [], lw=2, linestyle="--")
		axs.append(ax)
		lines.append(line)
		energys.append(energy)
		my_texts.append(my_text)
		shades.append(potential_shade)

	for i in range(9):
		axs[i].set_xlim(r_min, r_max)
		axs[i].set_ylim(-6, 6)

	def animate(i):
		for j in range(9):
			# shades[j] = axs[j].fill_between(rs, effective_potential(rs, j - 1, particle_object),  color="silver")
			shades[j].set_data(rs, effective_potential(rs, j - 1, particle_object))
			energys[j].set_data(rs, (0.1 + 0.01 * i) * np.ones_like(rs))
			my_texts[j].set_text("l = {0},E = {1:5.2f}meV".format(j, 0.1 + 0.01 * i))
			lines[j].set_data(rs, 1e3 * np.real(radial_schrodinger_equation(rs, dr, 0.1 + 0.01 * i, j, particle_object)) / np.sum(np.absolute(radial_schrodinger_equation(rs, dr, 0.1 + 0.01 * i, j, particle_object))))
		return lines + my_texts + shades + energys

	my_animation = animation.FuncAnimation(fig, animate, frames=500, interval=2, blit=True, repeat=True)
	# plt.show()
	my_animation.save('radial_wave_function_xeon.mp4', writer=writer)

# kr = Particle("Krypton", 83.798, 5.90, 3.57, [(321, 5), (550, 10), (755, 15)], [(0.50, 0.02), (1.59, 0.06), (2.94, 0.12)])
# xe = Particle("Xeon", 131.293, 6.65, 3.82, [(362.5, 12), (592, 20), (790, 25), (980, 25)], [(0.68, 0.04), (1.80, 0.12), (3.21, 0.20), (4.94, 0.25)])
# animation_plot(kr)
# animation_plot(xe)
