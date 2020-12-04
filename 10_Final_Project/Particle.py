class Particle(object):
	"""
	This data object is to store the basic information for the scattering calculation
	\n name is the particle name, string type
	\n mass is the atomic mass in unit of dalton(u)
	\n eps is the one of two parameters in the LJ potential and its unit is meV
	\n rho is the other parameters in the LJ potential and its unit is angstrom
	\n peak_v_info is the experimental data for the peak position(m/s) and its error, a list of tuples
	\n peak_e_info is the experimental data for the peak position(meV) and its error, a list of tuples
	"""
	def __init__(self, name, mass, eps, rho, peak_v_info, peak_e_info):
		self.name = name
		self.mass = mass
		self.eps = eps
		self.rho = rho
		self.peak_v_info = peak_v_info
		self.peak_e_info = peak_e_info

