import numpy as np
import scipy.fft
import time
import matplotlib.pyplot as plt


def print_result(times, error, name):
	fr = open("times_{0}.dat".format(name), 'w')
	fv = open("error_{0}.dat".format(name), 'w')

	for j in range(times.shape[0]):
		fr.write(str(times[j]) + '\n')
		fv.write(str(error[j]) + '\n')
	fr.write('\n')
	fv.write('\n')


def my_dft(y_array):
	N = np.size(y_array)

	n_range = np.arange(N, dtype='complex')
	"""""""""""""""""""""""""""""""""""""""""""""
	More Space Needed, N x N. 
	When N > 10^4, this algorithm will raise problems
	"""""""""""""""""""""""""""""""""""""""""""""
	# exp_range = np.repeat([n_range], N, axis=0)
	# exp_range = exp_range.astype('complex')
	#
	# for row_index in range(N):
	# 	exp_range[row_index, :] = np.exp(-2.0j * np.pi * exp_range[row_index, :] * row_index / N)
	# c_vec = np.dot(exp_range, y_array)
	"""""""""""""""""""""""""""""""""""""""""""""
	Less Space Needed, N elements each loop
	Still works for 2^18, do not try larger value
	"""""""""""""""""""""""""""""""""""""""""""""
	c_vec = np.zeros(N, dtype='complex')
	for k in range(N):
		exp_range = np.exp(-2.0j * np.pi * k * n_range / N, dtype='complex')
		c_vec[k] = np.dot(exp_range, y_array)

	return c_vec
	

def my_fft(y_array):
	# https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
	N = len(y_array)
	if N <= 32:
		return my_dft(y_array)
	else:
		E = my_fft(y_array[::2])
		O = my_fft(y_array[1::2])
		k = np.arange(N)
		C = np.exp(-2j * np.pi * k / N)

		Y = np.concatenate([E + C[:N // 2] * O, E + C[N // 2:] * O])
		return Y


def my_benchmark_dft(y_array):

	bi_c_vec = scipy.fft.fft(y_array)
	start_time = time.time()
	my_c_vec = my_dft(y_array)
	func_time = time.time() - start_time
	abs_error_mean = np.mean(np.abs(my_c_vec - bi_c_vec) / np.abs(bi_c_vec))
	if not np.allclose(my_c_vec, bi_c_vec):
		print("---My DFT NOT working---")
		return -1
	if func_time >= 1e-4:
		return func_time, abs_error_mean
	else:
		start_time = time.time()
		for n in range(10000):
			_ = scipy.fft.fft(y_array)
		func_time = (time.time() - start_time) / 10000
		return func_time, abs_error_mean


def benchmark_plot_dft():
	N_points = 17
	low_bound = 1
	upper_bound = 17
	time_range = np.logspace(low_bound, upper_bound, N_points, base=2.0)
	time_list = []
	error_list = []
	for array_size in time_range:
		print(array_size)
		bm_time, bm_error = my_benchmark_dft(np.arange(array_size))
		if bm_time == -1:
			return
		time_list.append(bm_time)
		error_list.append(bm_error)

	time_list = np.array(time_list).reshape(N_points)
	error_list = np.array(error_list).reshape(N_points)
	print_result(time_list, error_list, "DFT")
	# cmplx_fit = np.polyfit(time_range[-2 * N_points // (upper_bound - low_bound):], time_list[-2 * N_points // (upper_bound - low_bound):], 2)
	# p = np.poly1d(cmplx_fit)

	fig = plt.figure(figsize=(8, 8))
	fig.suptitle("DFT Benchmark", fontsize=16)

	axis1 = fig.add_subplot(211)
	axis1.title.set_text('Time Cost of My DFT(log)')
	axis1.plot(time_range, time_list, 'bo', label="Samples")
	# axis1.plot(time_range[- 2 * N_points // (upper_bound - low_bound):], p(time_range[-2 * N_points // (upper_bound - low_bound):]), "r-", label=r"$\mathcal{O}(N^2)$")
	axis1.legend()
	axis1.set_xlabel("Vector Size: N [log]")
	axis1.set_ylabel("Time(s) [log]")
	axis1.set_xscale('log')
	axis1.set_yscale('log')
	axis1.grid(True)

	axis2 = fig.add_subplot(212)
	axis2.title.set_text('Error of My DFT(log)')
	axis2.plot(time_range, error_list, 'bo', label="Samples")
	axis2.set_xlabel("Vector Size: N [log]")
	axis2.set_ylabel("Absolute Error Mean [log]")
	axis2.set_xscale('log')
	axis2.set_yscale('log')
	axis2.grid(True)

	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	# plt.show()
	fig.savefig("DFT_Benchmark.pdf", format="pdf")
	print("DFT Done")


def my_benchmark_fft(y_array):
	bi_c_vec = scipy.fft.fft(y_array)
	start_time = time.time()
	my_c_vec = my_fft(y_array)
	func_time = time.time() - start_time
	abs_error_mean = np.mean(np.abs(my_c_vec - bi_c_vec) / np.abs(bi_c_vec))
	if not np.allclose(my_c_vec, bi_c_vec):
		print("---My DFT NOT working---")
		return -1
	if func_time >= 1e-4:
		return func_time, abs_error_mean
	else:
		start_time = time.time()
		for n in range(10000):
			_ = scipy.fft.fft(y_array)
		func_time = (time.time() - start_time) / 10000
		return func_time, abs_error_mean


def benchmark_plot_fft():
	N_points = 22
	low_bound = 1
	upper_bound = 22

	time_range = np.logspace(low_bound, upper_bound, N_points, base=2.0)
	time_list = []
	error_list = []
	for array_size in time_range:
		# print(array_size)
		bm_time, bm_error = my_benchmark_fft(np.arange(array_size))
		if bm_time == -1:
			return
		time_list.append(bm_time)
		error_list.append(bm_error)

	time_list = np.array(time_list).reshape(N_points)
	error_list = np.array(error_list).reshape(N_points)
	print_result(time_list, error_list, "FFT")

	# cmplx_fit = np.polyfit(time_range * np.log2(time_range), time_list, 1)


	fig = plt.figure(figsize=(8, 8))
	fig.suptitle("FFT Benchmark", fontsize=16)

	axis1 = fig.add_subplot(211)
	axis1.title.set_text("Time Cost of My FFT(log)")
	axis1.plot(time_range, time_list, 'bo', label="Samples")
	# axis1.plot(time_range, cmplx_fit[0] * time_range * np.log2(time_range) + cmplx_fit[1], 'r-', label=r"$\mathcal{O}(n\log n)$")
	axis1.legend()
	axis1.set_xlabel("Vector Size: N")
	axis1.set_ylabel("Time(s) [log]")
	# axis1.set_xscale('log')
	axis1.set_yscale('log')
	axis1.grid(True)

	axis2 = fig.add_subplot(212)
	axis2.title.set_text("Error of My FFT(log)")
	axis2.plot(time_range, error_list, 'bo')
	axis2.set_xlabel("Vector Size: N")
	axis2.set_ylabel("Absolute Error Mean")
	# axis2.set_xscale('log')
	axis2.set_yscale('log')
	axis2.grid(True)

	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	# plt.show()
	fig.savefig("FFT_Benchmark.pdf", format="pdf")
	print("FFT Done")


def built_in_comp():
	dft_time = np.loadtxt("times_DFT.dat")
	fft_time = np.loadtxt("times_FFT.dat")
	dft_N = dft_time.size
	fft_N = fft_time.size
	size_range = np.logspace(1, max(dft_N, fft_N), max(dft_N, fft_N), base=2.0)
	size_range_dft = np.logspace(1, dft_N, dft_N, base=2.0)
	size_range_fft = np.logspace(1, fft_N, fft_N, base=2.0)
	bi_fft_time = np.zeros(max(dft_N, fft_N))

	for i in range(0, len(size_range)):
		start_time = time.time()
		_ = scipy.fft.fft(np.arange(size_range[i]))
		trial_time = time.time() - start_time
		if trial_time >= 1e-4:
			bi_fft_time[i] = (time.time() - start_time)
		else:
			for n in range(10000):
				_ = scipy.fft.fft(np.arange(size_range[i]))
			bi_fft_time[i] = (time.time() - start_time) / 10000


	new_x = np.arange(100, 10e6, 100)
	dft_fit = np.polyfit(size_range_dft[-(dft_N-4):] * size_range_dft[-(dft_N-4):], dft_time[-(dft_N-4):], 1)
	new_dft = dft_fit[0] * new_x * new_x
	fft_fit = np.polyfit(size_range_fft[-(fft_N-4):] * np.log2(size_range_fft[-(fft_N-4):]), fft_time[-(fft_N-4):], 1)
	new_fft = fft_fit[0] * new_x * np.log2(new_x)
	bi_fft_fit = np.polyfit(size_range[-(fft_N-4):] * np.log2(size_range[-(fft_N-4):]), bi_fft_time[-(fft_N-4):], 1)
	new_bi_fft = bi_fft_fit[0] * new_x * np.log2(new_x)

	fig = plt.figure(figsize=(8, 8))
	axis1 = fig.add_subplot(111)
	axis1.title.set_text("Time Complexity Comparison and Curve Fitting")
	axis1.plot(size_range, bi_fft_time, 'ro', label="Built-in FFT")
	axis1.plot(new_x, new_bi_fft, 'r-', label=r"Built-in FFT $\mathcal{{O}}\mathrm{{(Nlog_{{}}(N))}}, K={0:.2}, b={1:.2}$".format(bi_fft_fit[0], bi_fft_fit[1]))

	axis1.plot(size_range_dft, dft_time, 'go', label="My DFT")
	axis1.plot(new_x, new_dft, 'g-', label=r"My DFT $\mathcal{{O}}\mathrm{{(N^2)}}, K={0:.2}, b={1:.2}$".format(dft_fit[0], dft_fit[1]))

	axis1.plot(size_range_fft, fft_time, 'bo', label="My FFT")
	axis1.plot(new_x, new_fft, 'b-', label=r"My FFT $\mathcal{{O}}\mathrm{{(Nlog_{{}}(N))}}, K={0:.2}, b={1:.2}$".format(fft_fit[0], fft_fit[1]))

	axis1.axhline(1.0, color='k', ls='--')
	idx_myDFT = np.argwhere(np.diff(np.sign(1 - new_dft))).flatten()
	idx_myFFT = np.argwhere(np.diff(np.sign(1 - new_fft))).flatten()
	print("The largest vector that I can transform within a second:\n\tWith DFT: {0}\n\tWith FFT: {1}".format(new_x[idx_myDFT][0], new_x[idx_myFFT[0]]))
	axis1.plot(new_x[idx_myDFT], new_dft[idx_myDFT], 'kX', markersize=12.0)
	axis1.plot(new_x[idx_myFFT], new_fft[idx_myFFT], 'kX', markersize=12.0)

	axis1.set_xscale("log")
	axis1.set_yscale("log")
	axis1.legend()
	# plt.show()
	fig.savefig("Time Complexity.pdf", format="pdf")


# benchmark_plot_dft()
# benchmark_plot_fft()
built_in_comp()
