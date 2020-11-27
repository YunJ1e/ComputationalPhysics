import numpy as np
import scipy.special


def phase_shift(rs, dr, psi, E, l):
	"""
	Calculate the phase shift given
	:param rs:
	:param dr:
	:param psi:
	:param Es: Energy
	:param l: Dimensionless
	:return:
	"""
	r_max = rs[-1]
	r1, r2 = r_max - dr, r_max
	u1, u2 = psi[-2], psi[-1]
	K = r1 * u2 / (r2 * u1)
	k = np.sqrt(6.12 * E)
	j1, j2 = scipy.special.spherical_jn(l, k * r1), scipy.special.spherical_jn(l, k * r2)
	n1, n2 = scipy.special.spherical_yn(l, k * r1), scipy.special.spherical_yn(l, k * r2)
	delta_l = np.arctan(((K * j1 - j2) / (K * n1 - n2)).real)

	return delta_l
