import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from Particle import Particle


def effective_mass(particle_object):
	"""
	Calculate the effective mass of the two-particle system
	:param particle_object:
	:return:
	"""
	particle_mass = particle_object.mass
	amu_in_kg = const.atomic_mass

	m_hydrogen = 1.008
	m_particle = particle_mass

	m_effective = amu_in_kg * m_particle * m_hydrogen / (m_particle + m_hydrogen)
	return amu_in_kg * m_hydrogen


def prefactor(particle_object):
	"""
	Calculate the prefactor 2 * m / hbar^2 in unit of (mev * rho^2)^-1
	:param particle_object: contains necessary parameters for the calculation
	:return: the prefactor in the unit of (mev * rho^2)^-1, for H-Kr, reference is 6.12 and this function gives 6.07
	"""

	particle_rho = particle_object.rho
	milli_ev_to_joule_ratio = const.e * 1e-3
	angstrom_ratio = 1e-10
	m_effective = effective_mass(particle_object)  # Mass in SI unit

	two_m_over_hbar_square = 2 * m_effective / (const.hbar * const.hbar)  # Mass in SI unit
	two_m_over_hbar_square *= ((particle_rho * angstrom_ratio)**2 * milli_ev_to_joule_ratio)
	return two_m_over_hbar_square


def lj_potential(r):
	"""
	The two-atom interaction potential for atoms is often modelled by LJ potential,
	:param r: position is in unit of rho
	:return: The dimensionless LJ potential
	"""
	r_6 = (1.0 / r) ** 6
	return r_6 ** 2 - 2 * r_6


def effective_potential(r, l, particle_object):
	"""
	The LJ interaction potential plus the centrifugal barrier of the radial Schrodinger equation
	:param r: position is in unit of rho
	:param l: Dimensionless
	:param particle_object: contains necessary parameters for the calculation
	:return: The effective potential in unit of meV
	"""
	# Magic number 6.12 in unit of (meV * rho^2)^-1 = (2m) / hbar^2
	eps = particle_object.eps
	two_m_hbar_square = prefactor(particle_object)
	return eps * lj_potential(r) + (l + 1) * l / two_m_hbar_square / (r ** 2)


def plot_effective_potential(particle_object, ls=np.arange(11)):
	"""
	Plot the effective potential for the LJ interaction for various l-values
	:param particle_object: contains necessary parameters for the calculation
	:param ls: Dimensionless
	:return: None
	"""
	# Create the internuclear distance grid in unit of rho
	r_min = 0.5
	r_max = 5.5
	r_steps = 1000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)
	# Plot
	fig = plt.figure(figsize=(8, 8))
	fig.add_subplot(1, 1, 1)

	for l in ls:
		plt.plot(rs, effective_potential(rs, l, particle_object), label=r"$\mathrm{{l}}$={0}".format(l))
	plt.title(r'H-{0} Effective potential'.format(particle_object.name))
	plt.ylim(-8, 8)
	plt.hlines(0, 0, r_max, color='k', linestyles='solid')
	plt.xlabel(r"$r[\rho]$", fontsize=12)
	plt.ylabel(r"$V_{eff}(r)$[meV]", fontsize=12)
	plt.legend()
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	# plt.show()
	fig.savefig("{0}EffectivePotential_Final.pdf".format(particle_object.name), format="pdf")


# kr = Particle("Krypton", 83.798, 5.90, 3.57, [(321, 5), (550, 10), (755, 15)], [(0.50, 0.02), (1.59, 0.06), (2.94, 0.12)])
# xe = Particle("Xeon", 131.293, 6.65, 3.82, [(362.5, 12), (592, 20), (790, 25), (980, 25)], [(0.68, 0.04), (1.80, 0.12), (3.21, 0.20), (4.94, 0.25)])
# plot_effective_potential(kr)
# plot_effective_potential(xe)


