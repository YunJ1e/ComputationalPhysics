import numpy as np
import matplotlib.pyplot as plt
from Numerov import radial_schrodinger_equation
from PhaseShift import phase_shift
from Potential import prefactor, effective_mass
from Particle import Particle


def test(particle_object):
	"""
	Calculate and plot the total cross section for certain particle
	:param particle_object: the data structure containing necessary information of the scattering particle
	:return: None
	"""
	eps = particle_object.eps
	two_m_hbar_square = prefactor(particle_object)
	eff_mass = effective_mass(particle_object)

	r_min = 0.5
	r_max = 5.5
	r_steps = 1000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)

	Es = np.linspace(0.4, 5.4, 200)
	delta_l = np.zeros_like(Es)
	total_delta_l = np.zeros_like(Es)
	ls = np.arange(0, 11, dtype=np.int8)

	fig = plt.figure(figsize=(12, 9))  # plot the calculated values
	fig.add_subplot(1, 1, 1)

	for l in ls:
		for i in range(np.size(Es)):
			k = two_m_hbar_square * Es[i]
			psi = radial_schrodinger_equation(rs, dr, Es[i], l, particle_object)
			delta_l[i] = 4.0 * np.pi * (2 * l + 1) * (np.sin(phase_shift(rs, dr, psi, Es[i], l, two_m_hbar_square)))**2 / k
			total_delta_l[i] += delta_l[i]
		# plt.plot(np.sqrt(2 * Es * const.e * 1e-3 / eff_mass), delta_l, label=r" l={0}".format(l))
		plt.plot(Es, delta_l, label=r" l={0}".format(l))
	plt.title(r'Total Cross Section(Hydrogen-{0})'.format(particle_object.name))
	plt.xlabel(r"Energy[meV]", fontsize=12)
	plt.ylabel(r"Total cross section[$\rho^2$]", fontsize=12)
	plt.xlim(left=0.0)
	plt.fill_between(Es, total_delta_l, color='silver', label=r"Total $\delta$")
	for peak in particle_object.peak_e_info:
		plt.axvspan(peak[0] - peak[1], peak[0] + peak[1], ymin=0.0)
	plt.plot(Es, total_delta_l, "--")
	plt.legend()
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	# plt.show()
	fig.savefig("{0}TotalCrossSection_Final.pdf".format(particle_object.name), format="pdf")


def test02(particle_object):
	"""
	Calculate and plot the phase shifts for certain particle
	:param particle_object: the data structure containing necessary information of the scattering particle
	:return: None
	"""
	two_m_hbar_square = prefactor(particle_object)
	r_min = 0.5
	r_max = 5.5
	r_steps = 1000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)

	Es = np.linspace(0.4, 5, 200)
	delta_l = np.zeros_like(Es)
	ls = np.arange(0, 9, dtype=np.int8)

	fig = plt.figure(figsize=(12, 9))  # plot the calculated values
	fig.add_subplot(1, 1, 1)

	for l in ls:
		for i in range(np.size(Es)):
			psi = radial_schrodinger_equation(rs, dr, Es[i], l, particle_object)
			delta_l[i] = phase_shift(rs, dr, psi, Es[i], l, two_m_hbar_square)
		delta_diff = np.diff(delta_l)
		flip_position = np.where(np.diff(np.sign(delta_l)))[0]
		discard_flag = False

		if np.size(flip_position) > 0:
			first_flip = flip_position[0]
			if delta_diff[first_flip] >= (np.pi / 2):
				delta_l[0: first_flip + 1] += np.pi
				if np.size(flip_position) >= 2:
					discard_flag = True
			elif delta_diff[first_flip] <= -(np.pi / 2):
				if np.size(flip_position) >= 2:
					delta_l[first_flip + 1:flip_position[1]+1] += np.pi
				else:
					delta_l[first_flip + 1:] += np.pi
					if np.size(flip_position) >= 2:
						discard_flag = True
			else:
				second_flip = flip_position[1]
				if delta_diff[second_flip] >= (np.pi / 2):
					delta_l[0: second_flip + 1] += np.pi
				elif delta_diff[second_flip] <= -(np.pi / 2):
					delta_l[second_flip + 1:] += np.pi
		if l in [4, 5, 6, 7]:
			if not discard_flag:
				plt.plot(Es, delta_l, label=r" l={0}".format(l), linewidth=6, linestyle="-")
			else:
				plt.plot(Es[:flip_position[2] + 1], delta_l[:flip_position[2] + 1], label=r" l={0}".format(l), linewidth=6, linestyle="-")
		else:
			if not discard_flag:
				plt.plot(Es, delta_l, label=r" l={0}".format(l), linewidth=3, linestyle="--")
			else:
				plt.plot(Es[:flip_position[2] + 1], delta_l[:flip_position[2] + 1], label=r" l={0}".format(l), linewidth=3, linestyle="--")

	plt.hlines(np.pi / 2, Es[0], Es[-1])
	for peak in particle_object.peak_e_info:
		plt.axvspan(peak[0] - peak[1], peak[0] + peak[1], ymin=0.0, color="grey", alpha=0.5)
	plt.title(r'Phase Shifts $\delta_{l}$')
	plt.xlabel(r"Energy[meV]", fontsize=12)
	plt.ylabel(r"$\delta_{l}[rad]$", fontsize=12)

	plt.legend()
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	# plt.show()
	fig.savefig("{0}PhaseShift.pdf".format(particle_object.name), format="pdf")


ar = Particle("Argon", 39.948, 4.16, 3.62, [(455, 10), (660, 20)], [(1.05, 0.05), (2.22, 0.12)])
kr = Particle("Krypton", 83.798, 5.90, 3.57, [(321, 5), (550, 10), (755, 15)], [(0.50, 0.02), (1.59, 0.06), (2.94, 0.12)])
xe = Particle("Xeon", 131.293, 6.305, 3.34, [(362.5, 12), (592, 20), (790, 25), (980, 25)], [(0.68, 0.04), (1.80, 0.12), (3.21, 0.20), (4.94, 0.25)])

# test(kr)
# test(xe)
# test02(xe)
# test02(kr)

