import numpy as np
import matplotlib.pyplot as plt
from Numerov import radial_schrodinger_equation
from PhaseShift import phase_shift


def test():

	r_min = 0.5
	r_max = 5
	r_steps = 1000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)

	Es = np.linspace(0.1, 3.5, 200)
	delta_l = np.zeros_like(Es)
	total_delta_l = np.zeros_like(Es)
	ls = np.arange(1, 11, dtype=np.int8)

	fig = plt.figure(figsize=(12, 9))  # plot the calculated values
	fig.add_subplot(1, 1, 1)

	for l in ls:
		for i in range(np.size(Es)):
			k = np.sqrt(6.12 * Es[i])
			psi = radial_schrodinger_equation(rs, dr, Es[i], l)
			delta_l[i] = 4.0 * np.pi * (2 * l + 1) * (np.sin(phase_shift(rs, dr, psi, Es[i], l)))**2 / k / k
			total_delta_l[i] += delta_l[i]
		plt.plot(Es, delta_l, label=r" l={0}".format(l))
	plt.title(r'Total Cross Section(H-Kr)')
	plt.xlabel(r"Energy[meV]", fontsize=12)
	plt.ylabel(r"Total cross section[$\rho^2$]", fontsize=12)

	plt.fill_between(Es, total_delta_l, color='silver', label=r"Total $\delta$")
	plt.plot(Es, total_delta_l, "--")
	plt.legend()
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.show()
# fig.savefig("EffectivePotential.pdf", format="pdf")


def test02():

	r_min = 0.5
	r_max = 5
	r_steps = 1000
	rs, dr = np.linspace(r_min, r_max, r_steps, retstep=True)

	Es = np.linspace(0.1, 3.5, 200)
	delta_l = np.zeros_like(Es)
	total_delta_l = np.zeros_like(Es)
	ls = np.arange(4, 7, dtype=np.int8)

	fig = plt.figure(figsize=(12, 9))  # plot the calculated values
	fig.add_subplot(1, 1, 1)

	for l in ls:
		for i in range(np.size(Es)):
			k = np.sqrt(6.12 * Es[i])
			psi = radial_schrodinger_equation(rs, dr, Es[i], l)
			delta_l[i] = phase_shift(rs, dr, psi, Es[i], l)
			# total_delta_l[i] += delta_l[i]
		plt.plot(Es, delta_l, label=r" l={0}".format(l))
	plt.xscale('log')
	plt.title(r'Total Cross Section(H-Kr)')
	plt.xlabel(r"Energy[meV]", fontsize=12)
	plt.ylabel(r"Total cross section[$\rho^2$]", fontsize=12)

	# plt.fill_between(Es, total_delta_l, color='silver', label=r"Total $\delta$")
	# plt.plot(Es, total_delta_l, "--")
	plt.legend()
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.show()

test()
# test02()

# if __name__ == '__main__':
# 	print_hi('PyCharm')
