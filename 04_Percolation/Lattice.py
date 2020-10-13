import numpy as np
import matplotlib.pyplot as plt
import copy


def init_square_lattice(L, p):

	grid = np.random.random([L, L])
	grid = np.ceil(grid - (1.0 - p))

	# To check the random numbers are valid
	# print(np.count_nonzero(grid) / L**2)
	# row, col = np.where(grid == 1.0)
	# plt.title("Example Grid")
	# plt.imshow(grid, cmap='gray_r')
	# plt.colorbar(ticks=range(2), label='digit value')
	# plt.show()
	return grid


def occupied_lattice(grid):
	row, col = np.where(grid == 1.0)
	if not np.size(row) == np.size(row):
		print("---Occupied Lattice Size Does Not Match---")
		return -1
	occupied_index = []
	for i in range(np.size(row)):
		occupied_index.append((row[i], col[i]))
	return occupied_index


def find(grid, current_row, current_col, ori_up, ori_left):
	if current_row + ori_up < 0 or current_col + ori_left < 0:
		return False
	if grid[current_row + ori_up][current_col + ori_left] == 0.0:
		return False
	return True


def color_lattice(grid, n, type):
	# palette = copy.copy(plt.get_cmap('viridis_r'))
	# palette.set_under('white', 1.0)
	discretize_cmap = copy.copy(plt.cm.get_cmap("jet_r", n))
	discretize_cmap.set_under('white', 1.0)
	fig = plt.figure(figsize=(6, 6))
	axis1 = fig.add_subplot(111)
	# img = axis1.imshow(grid, cmap="viridis_r",aspect='auto')
	img = axis1.imshow(grid, cmap=discretize_cmap, vmin=0.5, vmax=n+0.5, aspect='auto')
	# img = axis1.imshow(grid, cmap=discretize_cmap)
	axis1.title.set_text("{0} Example P: 0.58".format(type))
	fig.colorbar(img, ticks=range(1, n+1), label='Label value')
	fig.savefig("{0}_Example.pdf".format(type), format="pdf")
	# plt.show()


def hoshen_kopelman_label(N, p):

	myGrid = init_square_lattice(N, p)
	# myGrid = np.array([[1, 0, 1, 1, 0, 1, 1, 1],
	#                    [1, 0, 0, 0, 1, 1, 1, 0],
	#                    [1, 0, 1, 0, 1, 0, 0, 1],
	#                    [1, 1, 1, 1, 1, 0, 1, 0],
	#                    [0, 0, 0, 0, 0, 0, 0, 0],
	#                    [0, 0, 0, 0, 0, 0, 0, 0],
	#                    [0, 0, 0, 0, 0, 0, 0, 0],
	#                    [0, 0, 0, 0, 0, 0, 0, 0]])
	occupiedGrid = occupied_lattice(myGrid)
	proper_label = np.arange(0, N*N)

	num_of_label = 0
	for i in range(N):
		for j in range(N):
			if (i, j) in occupiedGrid:
				# Left and up elements are not occupied
				if not (find(myGrid, i, j, 0, -1) or find(myGrid, i, j, -1, 0)):
					# Use a new label
					num_of_label += 1
					myGrid[i][j] = num_of_label
				# Up occupied but left does not
				elif find(myGrid, i, j, -1, 0) and not find(myGrid, i, j, 0, -1):
					myGrid[i][j] = myGrid[i - 1][j]
				# Left occupied but up does not
				elif find(myGrid, i, j, 0, -1) and not find(myGrid, i, j, -1, 0):
					myGrid[i][j] = myGrid[i][j - 1]
				# Cluster Collision
				else:
					# Find the minimum label
					small_label = int(min(myGrid[i - 1][j], myGrid[i][j - 1]))
					large_label = int(max(myGrid[i - 1][j], myGrid[i][j - 1]))
					myGrid[i][j] = small_label
					proper_label[proper_label == proper_label[large_label]] = proper_label[small_label]

	# Right now we have "num_of_label" labels assigned to clusters, but due to the collision, some of the
	# numbers might not be used anymore, because every collision will result in one label un-used.
	# Here only keep the elements (0 ~ "number_of_label") in the old "proper_label" and using unique assigning
	# them new labels(0 ~ "number_of_clusters")
	# print(myGrid)
	_, new_proper_label = np.unique(proper_label[:num_of_label+1], return_inverse=True)

	for (i, j) in occupiedGrid:
		myGrid[i][j] = int(new_proper_label[int(myGrid[i][j])])
	# print(myGrid)

	return myGrid, np.max(new_proper_label)


def is_percolating(hk_matrix):
	for num in hk_matrix[0]:
		# print(num)
		if num in hk_matrix[-1] and num > 0:
			return num
	return -1


def percolation_threshold(N, start_prob, end_prob, prob_step=0.01, Nrep=100):
	probs = np.arange(start_prob, end_prob + prob_step, prob_step)
	# print(probs)
	prob_percolation = np.zeros(np.size(probs))
	print("---N: {1} Repetition: {0}---".format(Nrep, N))
	for i in range(np.size(probs)):
		# print("Percolating: p = {0:4.2f}".format(probs[i]))
		for _ in range(Nrep):
			hk_matrix, n = hoshen_kopelman_label(N, probs[i])
			if is_percolating(hk_matrix) != -1:
				prob_percolation[i] += 1
	prob_percolation = prob_percolation / Nrep
	print_result(prob_percolation, Nrep, N)


def print_result(probs, Nrep, N):
	f1 = open("P(p)_{0}_{1}.dat".format(Nrep, N), 'w')
	for j in range(probs.shape[0]):
		f1.write(str(probs[j]) + '\n')
	f1.write('\n')


def plot_prob(N, start_prob, end_prob, prob_step=0.01, Nrep=100):

	a = np.loadtxt("P(p)_{0}_{1}.dat".format(Nrep, N))
	fig = plt.figure(figsize=(8, 8))

	axis1 = fig.add_subplot(111)
	axis1.title.set_text("A {1}x{1} Percolated Cluster P(p)  Reps:{0}".format(Nrep, N))
	axis1.plot(np.arange(start_prob, end_prob + prob_step, prob_step), a, '-',linewidth=2.0)
	axis1.axvline(0.59, color='k', ls='--')
	axis1.axhline(0.50, color='b', ls='-.')
	axis1.set_xlabel("The probability of lattice element being 1: p")
	axis1.set_ylabel("The probability of being a percolating cluster: P")

	fig.savefig("P(p)_{0}_{1}.pdf".format(Nrep, N), format="pdf")


def oneplot(s_prob, e_prob, prob_step=0.01):
	Ns = [5, 10, 15, 20, 25, 30, 35, 40]
	Nreps = [100]
	fig = plt.figure(figsize=(8, 8))
	axis1 = fig.add_subplot(111)
	axis1.title.set_text("A Percolated Cluster P(p)")
	axis1.axvline(0.59, color='k', ls='--', label="x=0.59")
	axis1.axhline(0.50, color='b', ls='-.', label="y=0.50")
	axis1.set_xlabel("The probability of lattice element being 1: p")
	axis1.set_ylabel("The probability of being a percolating cluster: P")
	for N in Ns:
		for rep in Nreps:
			a = np.loadtxt("P(p)_{0}_{1}.dat".format(rep, N))
			axis1.plot(np.arange(s_prob, e_prob + prob_step, prob_step), a, label="reps={0}, N={1}".format(rep, N))
	axis1.legend()
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	fig.savefig("OnePlot.pdf", format="pdf")


def problem03(recalculate=False):
	Ns = [5, 10, 15, 20, 25, 30, 35, 40]
	s_prob = 0.4
	e_prob = 0.8
	Nreps = [100]

	for N in Ns:
		for Nrep in Nreps:
			if recalculate:
				percolation_threshold(N, s_prob, e_prob, Nrep=Nrep)
			plot_prob(N, s_prob, e_prob, Nrep=Nrep)
	oneplot(s_prob, e_prob)


def problem02():
	perc_flag = False
	nonperc_flag = False
	while (not perc_flag) or (not nonperc_flag):
		aMatrix, n = hoshen_kopelman_label(10, 0.58)
		if is_percolating(aMatrix) != -1 and not perc_flag:
			color_lattice(aMatrix, n, "Percolating")
			perc_flag = True
		elif is_percolating(aMatrix) == -1 and not nonperc_flag:
			color_lattice(aMatrix, n, "Non-Percolating")
			nonperc_flag = True


problem02()
problem03()



