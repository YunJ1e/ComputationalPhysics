import numpy as np
import matplotlib.pyplot as plt
import time

def init_square_lattice(L):
	grid = np.random.randint(2, size=(L, L))
	grid[grid == 0] = - 1

	# grid = np.random.random([L, L])
	# grid = np.ceil(grid - (1.0 - p))
	# print(grid)
	# plt.title("Example Grid")
	# plt.imshow(grid, cmap='bwr')
	# plt.colorbar(ticks=range(-1, 2), label='digit value')
	# plt.show()
	return grid


def periodic_bc(L, index):
	if index < 0:
		return L - 1
	elif index > L - 1:
		return 0
	else:
		return index


def near_neighbor_index(L, position):
	row, col = position[0], position[1]
	up, down = periodic_bc(L, row - 1), periodic_bc(L, row + 1)
	left, right = periodic_bc(L, col - 1), periodic_bc(L, col + 1)

	return [(up, col), (down, col), (row, left), (row, right)]


def near_neighbor_sum(lattice, row_index, col_index):
	j = -1.0
	L = lattice.shape[0]
	result = 0
	indices = near_neighbor_index(L, (row_index, col_index))
	for index in indices:
		result += j * lattice[index[0]][index[1]] * lattice[row_index][col_index]
	return result


def sum_nn(lattice):
	# print(lattice)
	# new_lattice = np.zeros(lattice.shape)
	config_energy = 0.0
	L = lattice.shape[0]
	for i in range(L):
		for j in range(L):
			config_energy += near_neighbor_sum(lattice, i, j)
			# print(near_neighbor_sum(lattice, i, j), end=" ")
		# print()
	return config_energy / 2


def spin_sum(lattice):
	return np.sum(lattice, dtype=np.int16)


def flip_one_spin(lattice):
	L = lattice.shape[0]
	# print(L)
	flipped_index = np.random.randint(L, size=2)
	# print(flipped_index)
	row_index, target_index = flipped_index[0], flipped_index[1]
	lattice[row_index][target_index] *= -1

	return flipped_index


def single_spin_metropolis(lattice, beta):
	# beta = 1/kT and k = 1
	old_ener = sum_nn(lattice)

	# The original lattice is flipped and the flipped indices are returned in case we need to flip back
	flipped_index = flip_one_spin(lattice)
	new_ener = sum_nn(lattice)
	dE = new_ener - old_ener
	# 0~1 random number to simulate the probability
	a = np.random.random()

	# Only at dE > 0 and also the random number greater than exp calculation,
	# we need to keep the original config(flip back)
	if dE > 0 and a > np.exp(-beta * dE):
		# Only at this situation, we need to flip the spin back to original
		row_index, col_index = flipped_index[0], flipped_index[1]
		lattice[row_index][col_index] *= -1


def print_result(nt, energy, magnetization, specificheat, magsuscept):
	fe = open("Energy.dat", 'w')
	fm = open("Magnetization.dat", 'w')
	fc = open("SpecificHeat.dat", 'w')
	fx = open("MagSuscept.dat", 'w')

	for j in range(nt):
		fe.write(str(energy[j]) + '\n')
		fm.write(str(magnetization[j]) + '\n')
		fc.write(str(specificheat[j]) + '\n')
		fx.write(str(magsuscept[j]) + '\n')

	fe.write('\n')
	fm.write('\n')
	fc.write('\n')
	fx.write('\n')


def print_fine_result(nt, energy, magnetization, specificheat, magsuscept):
	fe = open("Energy_Fine.dat", 'w')
	fm = open("Magnetization_Fine.dat", 'w')
	fc = open("SpecificHeat_Fine.dat", 'w')
	fx = open("MagSuscept_Fine.dat", 'w')

	for j in range(nt):
		fe.write(str(energy[j]) + '\n')
		fm.write(str(magnetization[j]) + '\n')
		fc.write(str(specificheat[j]) + '\n')
		fx.write(str(magsuscept[j]) + '\n')

	fe.write('\n')
	fm.write('\n')
	fc.write('\n')
	fx.write('\n')


def print_result_general(file_name, variable_name):
	fe = open(file_name, 'w')
	N = variable_name.shape[0]

	for j in range(N):
		fe.write(str(variable_name[j]) + '\n')
	fe.write('\n')
	fe.close()


def thermalize(original_lattice, recalculate=False, draw=True):
	L = original_lattice.shape[0]
	print(L)
	mcSteps = 50000
	# Only copy the value of the lattice, because we are studying the same configuration at different KT
	lattice = np.copy(original_lattice)
	Ts = [0.5, 2, 4]
	# The extra one is for recording the initial energy and spin sum
	E, M = np.zeros(mcSteps + 1), np.zeros(mcSteps + 1)
	n1 = 1.0 / (L * L)

	if recalculate:
		for KTemp in Ts:
			print("Temperature:", KTemp)
			# Record the initial value(also a way to make sure the copy is working)
			E[0] = sum_nn(lattice)
			M[0] = spin_sum(lattice)
			for i in range(1, len(E)):
				single_spin_metropolis(lattice, 1/KTemp)
				E[i] = sum_nn(lattice)
				M[i] = spin_sum(lattice)
			fe = open("EnergyVsMCSteps_{0}_{1}.dat".format(KTemp, L), 'w')
			fm = open("MagnetVsMCSteps_{0}_{1}.dat".format(KTemp, L), 'w')
			for step in range(mcSteps + 1):
				fe.write(str(E[step]) + '\n')
				fm.write(str(M[step]) + '\n')
			fe.write('\n')
			fm.write('\n')
			fe.close()
			fm.close()
			# Refresh the lattice with the original lattice
			lattice = np.copy(original_lattice)

	if draw:
		fig = plt.figure(figsize=(9, 6))  # plot the calculated values
		fig.suptitle("Physical Properties Versus MC Steps @ Different T", fontsize=20)
		fig.add_subplot(2, 1, 1)
		for T in Ts:
			plt.plot(range(mcSteps+1), np.loadtxt("EnergyVsMCSteps_{0}_{1}.dat".format(T, L)) * n1, label="T = {0}".format(T))
		plt.legend()
		plt.ylabel("Energy", fontsize=16)
		plt.axis('tight')

		fig.add_subplot(2, 1, 2)
		for T in Ts:
			plt.plot(range(mcSteps+1), np.abs(np.loadtxt("MagnetVsMCSteps_{0}_{1}.dat".format(T, L))) * n1, label="T = {0}".format(T))
		plt.legend()
		plt.ylabel("Magnetization", fontsize=16)

		plt.tight_layout()
		fig.subplots_adjust(top=0.88)
		# plt.show()
		fig.savefig("PropertiesVSMCsteps.pdf", format="pdf")


def mc_mearsurement(original_lattice, recalculate=False, draw=True):
	nt = 11
	N = 8
	eqSteps = 30000
	mcSteps = 30000
	T = np.linspace(1, 4, nt)
	T_fine = np.linspace(2.1, 2.3, nt)
	E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
	n1, n2 = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)

	if recalculate:
		for tt in range(nt):
			print(T[tt])
			E1 = M1 = E2 = M2 = 0
			config = np.copy(original_lattice)
			iT = 1.0 / T[tt]
			iT2 = iT * iT

			for i in range(eqSteps):
				single_spin_metropolis(config, iT)

			for i in range(mcSteps):
				single_spin_metropolis(config, iT)
				Ene = sum_nn(config)
				Mag = spin_sum(config)

				E1 = E1 + Ene
				M1 = M1 + Mag
				M2 = M2 + Mag * Mag
				E2 = E2 + Ene * Ene

			E[tt] = n1 * E1
			M[tt] = n1 * M1
			C[tt] = (n1 * E2 - n2 * E1 * E1) * iT2
			X[tt] = (n1 * M2 - n2 * M1 * M1) * iT
		print_result(nt, E, M, C, X)

		for tt in range(nt):
			print(T_fine[tt])
			E1 = M1 = E2 = M2 = 0
			config = np.copy(original_lattice)
			iT = 1.0 / T_fine[tt]
			iT2 = iT * iT

			for i in range(eqSteps):
				single_spin_metropolis(config, iT)

			for i in range(mcSteps):
				single_spin_metropolis(config, iT)
				Ene = sum_nn(config)
				Mag = spin_sum(config)

				E1 = E1 + Ene
				M1 = M1 + Mag
				M2 = M2 + Mag * Mag
				E2 = E2 + Ene * Ene

			E[tt] = n1 * E1
			M[tt] = n1 * M1
			C[tt] = (n1 * E2 - n2 * E1 * E1) * iT2
			X[tt] = (n1 * M2 - n2 * M1 * M1) * iT

		print_fine_result(nt, E, M, C, X)

	if draw:
		fig = plt.figure(figsize=(9, 6))  # plot the calculated values
		fig.suptitle("Physical Properties Versus Temperatures", fontsize=20)

		fig.add_subplot(2, 2, 1)
		plt.scatter(T, np.abs(np.loadtxt("Magnetization.dat")), s=10, label="Coarse Grid")
		plt.scatter(T_fine, np.abs(np.loadtxt("Magnetization_Fine.dat")), s=10, label="Fine Grid 2.1~2.3")
		plt.ylabel("Magnetization", fontsize=16)
		plt.legend()

		fig.add_subplot(2, 2, 2)
		plt.scatter(T, np.loadtxt("MagSuscept.dat"), s=10, label="Coarse Grid")
		plt.scatter(T_fine, np.loadtxt("MagSuscept_Fine.dat"), s=10, label="Fine Grid 2.1~2.3")
		plt.ylabel("Magnetic Susceptibility", fontsize=16)
		plt.legend()

		fig.add_subplot(2, 2, 3)
		plt.scatter(T, np.loadtxt("Energy.dat"), s=10, label="Coarse Grid")
		plt.scatter(T_fine, np.loadtxt("Energy_Fine.dat"), s=10, label="Fine Grid 2.1~2.3")
		plt.ylabel("Energy", fontsize=16)
		plt.legend()

		fig.add_subplot(2, 2, 4)
		plt.scatter(T, np.loadtxt("SpecificHeat.dat"), s=10, label="Coarse Grid")
		plt.scatter(T_fine, np.loadtxt("SpecificHeat_Fine.dat"), s=10, label="Fine Grid 2.1~2.3")
		plt.ylabel("Specific Heat", fontsize=16)
		plt.legend()

		plt.tight_layout()
		fig.subplots_adjust(top=0.88)
		# plt.show()
		fig.savefig("PropertiesVsTemps.pdf", format="pdf")


def binder_cumulant(T, original_lattice, recalculate=False):
	nt = T.shape[0]
	N = original_lattice.shape[0]

	eqSteps = 40000
	mcSteps = 50000

	binder_cumulants, Mag2, Mag4 = np.zeros(nt), np.zeros(nt), np.zeros(nt)
	n1 = 1.0 / (N * N)
	n2 = 1.0 / mcSteps

	if recalculate:
		for tt in range(nt):
			print(T[tt])
			M2 = M4 = 0
			config = np.copy(original_lattice)
			iT = 1.0 / T[tt]

			for i in range(eqSteps):
				single_spin_metropolis(config, iT)

			for i in range(mcSteps):
				single_spin_metropolis(config, iT)
				Mag = spin_sum(config) * n1

				M2 = M2 + Mag * Mag
				M4 = M4 + Mag * Mag * Mag * Mag
			# print(M2)
			# print(M4 * n2, (M2 * n2)**2.0)
			binder_cumulants[tt] = 1 - (M4 / (M2 * M2 * n2) / 3.0)

		print_result_general("Binder_Cumulant_{0}.dat".format(N), binder_cumulants)


def binder_cumulant_intersection(recalculate=False, draw=True):

	lattice_sizes = [4, 6, 8, 10]
	# lattice_sizes = [8]
	nt = 10
	T = np.linspace(1.8, 2.7, nt)

	if recalculate:
		for lattice_size in lattice_sizes:

			print("Lattice Size: {0} Starts...".format(lattice_size))
			start_time = time.time()
			mygrid = init_square_lattice(lattice_size)
			binder_cumulant(T, mygrid, recalculate)
			print("Lattice Size: {0} Done... Time：{1:.2f}s".format(lattice_size, time.time()-start_time))

	if draw:
		fig = plt.figure(figsize=(9, 6))
		fig.suptitle("Binder Cumulant Versus Temperatures", fontsize=20)
		fig.add_subplot(1, 1, 1)
		for lattice_size in lattice_sizes:
			plt.plot(T, np.loadtxt("Binder_Cumulant_{0}.dat".format(lattice_size)), label="L = {0}".format(lattice_size), linestyle='-',  marker='o')
			# plt.yscale('log')
		plt.xlabel("Temperature (T)", fontsize=16)
		plt.ylabel("Binder Cumulant", fontsize=16)
		plt.hlines(2 / 3, np.min(T), np.max(T), linestyle='--', label="Maximum: 2/3")
		plt.vlines(2.27, 0, 2 / 3, linestyle='-.', label=r"$T_{c} \sim 2.27$")
		plt.legend()
		plt.tight_layout()
		fig.subplots_adjust(top=0.88)
		# plt.show()
		fig.savefig("BinderCumulantVsTemps.pdf", format="pdf")


"""""""""""""""""""""""""""""""""""""""""""""""
	Section 01 
	The Properties V.s MC steps
"""""""""""""""""""""""""""""""""""""""""""""""
# mygrid = init_square_lattice(16)
# thermalize(mygrid, recalculate=True)

"""""""""""""""""""""""""""""""""""""""""
	Section 02 
	The property V.s Temperature and 
	the Phase Transition
"""""""""""""""""""""""""""""""""""""""""
# mc_mearsurement(mygrid, recalculate=False)

"""""""""""""""""""""""""""""""""""""""""
	Section 03
	Binder Cumulant
"""""""""""""""""""""""""""""""""""""""""
binder_cumulant_intersection(recalculate=False)







