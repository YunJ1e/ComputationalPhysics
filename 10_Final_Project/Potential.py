import numpy as np
import matplotlib.pyplot as plt


def lj_potential(r, eps=5.9):
	"""
	The two-atom interaction potential for atoms is often modelled by LJ potential,
	:param r: position is in unit of rho
	:param eps: LJ parameters(5.9 meV for H-Kr)
	:return: The LJ potential in unit of meV
	"""
	r_6 = (1.0 / r) ** 6
	return eps * (r_6 ** 2 - 2 * r_6)


def effective_potential(r, l):
	"""
	The LJ interaction potential plus the centrifugal barrier of the radial Schrodinger equation
	:param r: position is in unit of rho
	:param l: Dimensionless
	:return: The effective potential in unit of meV
	"""
	return lj_potential(r) + (l + 1) * l * 1.0 / 6.12 / (r ** 2)


def plot_effective_potential(ls=(0, 4, 5, 6)):
	"""
	Plot the effective potential for the LJ interaction for various l-values
	:param ls: Dimensionless
	:return: None
	"""
	r_min = 0.5
	r_max = 5
	r_steps = 10000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)
	fig = plt.figure(figsize=(5, 5))  # plot the calculated values

	fig.add_subplot(1, 1, 1)
	for l in ls:
		plt.plot(rs, effective_potential(rs, l), label=r" l={0}".format(l))
	plt.title(r'Effective potential')
	plt.ylim(-6, 6)
	plt.xlabel(r"$r[\rho]$", fontsize=12)
	plt.ylabel(r"$V_{eff}(r)$[meV]", fontsize=12)
	plt.legend()

	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	# plt.show()
	fig.savefig("EffectivePotential.pdf", format="pdf")


# plot_effective_potential()
