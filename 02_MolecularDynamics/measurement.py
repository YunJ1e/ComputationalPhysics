import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

solid_parameter = {"Nx": 6, "Ny": 6, "L": 6, "Nstep": int(6e4), "dt": 1e-3, "Method": "Verlet", "Phase": "Solid"}
fluid_parameter = {"Nx": 6, "Ny": 6, "L": 12, "Nstep": int(6e4), "dt": 1e-3, "Method": "Verlet", "Phase": "Fluid"}
testa = np.loadtxt("positions_Verlet_Solid_6.dat")
testb = np.loadtxt("positions_Verlet_Fluid_12.dat")
testa_x = np.reshape(testa[:, 0], (36, -1))
testa_y = np.reshape(testa[:, 1], (36, -1))
testb_x = np.reshape(testb[:, 0], (36, -1))
testb_y = np.reshape(testb[:, 1], (36, -1))

t_therm_solid = 10.0
t_therm_fluid = t_therm_solid


def measurement(parameters, t_therm, rxlog, rylog):
	dt = parameters['dt']
	t_total = parameters['Nstep'] * dt
	N = parameters['Nx'] * parameters['Ny']
	t_range = np.arange(0.0, t_total, dt)
	t_range_therm = np.arange(t_therm, t_total, dt)
	diffusion = []
	rmsd = []

	for j in range(len(t_range)):
		dx = rxlog[:, j] - rxlog[:, 0]
		dy = rylog[:, j] - rylog[:, 0]
		r = np.sqrt(dx ** 2 + dy ** 2)
		rmsd_value = (1 / N) * np.sum(r)
		rmsd.append(rmsd_value)
		diffusion.append(rmsd_value / 4 * t_range[j])


	coeff1 = np.polyfit(t_range_therm, rmsd[-len(t_range_therm):], 1)
	coeff2 = np.polyfit(t_range_therm, diffusion[-len(t_range_therm):], 1)

	return t_range, t_range_therm, rmsd, diffusion, coeff1, coeff2


"""""""""
Measurement I
"""""""""
t_range_solid, t_range_therm_solid, rmsd_solid, diffusion_solid, coeff1_solid, coeff2_solid =\
	measurement(solid_parameter, t_therm_solid, testa_x, testa_y)
t_range_fluid, t_range_therm_fluid, rmsd_fluid, diffusion_fluid, coeff1_fluid, coeff2_fluid = \
	measurement(fluid_parameter, t_therm_fluid, testb_x, testb_y)

# https://stackoverflow.com/questions/17788685/python-saving-multiple-figures-into-one-pdf-file
fig = plt.figure(figsize=(8,8))
fig.suptitle("Measurement I", fontsize=16)

axis1 = fig.add_subplot(221)
axis1.title.set_text('RMSD Solid')
axis1.plot(t_range_solid, rmsd_solid, 'bo')
axis1.plot(t_range_therm_solid, t_range_therm_solid * coeff1_solid[0] + coeff1_solid[1], 'k-', label="k={0:3f}, b={1:3f}".format(coeff1_solid[0], coeff1_solid[1]))
axis1.legend()
axis1.set_xlabel("Time(s)")
axis1.set_ylabel("RMSD Value")

axis2 = fig.add_subplot(223)
axis2.title.set_text('Self-Diffusion Coefficient Solid')
axis2.plot(t_range_solid, diffusion_solid, 'bo')
axis2.plot(t_range_therm_solid, t_range_therm_solid * coeff2_solid[0] + coeff2_solid[1], 'k-', label="k={0:3f}, b={1:3f}".format(coeff2_solid[0], coeff2_solid[1]))
axis2.legend()
axis2.set_xlabel("Time(s)")
axis2.set_ylabel("Self-Diffusion Coefficient Value")

axis3 = fig.add_subplot(222)
axis3.title.set_text('RMSD Fluid')
axis3.plot(t_range_fluid, rmsd_fluid, 'bo')
axis3.plot(t_range_therm_fluid, t_range_therm_fluid * coeff1_fluid[0] + coeff1_fluid[1], 'k-', label="k={0:3f}, b={1:3f}".format(coeff1_fluid[0], coeff1_fluid[1]))
axis3.legend()
axis3.set_xlabel("Time(s)")
axis3.set_ylabel("RMSD Value")

axis4 = fig.add_subplot(224)
axis4.title.set_text('Self-Diffusion Coefficient Fluid')
axis4.plot(t_range_fluid, diffusion_fluid, 'bo')
axis4.plot(t_range_therm_fluid, t_range_therm_fluid * coeff2_fluid[0] + coeff2_fluid[1], 'k-', label="k={0:3f}, b={1:3f}".format(coeff2_fluid[0], coeff2_fluid[1]))
axis4.legend()
axis4.set_xlabel("Time(s)")
axis4.set_ylabel("Self-Diffusion Coefficient Value")

plt.tight_layout()
fig.subplots_adjust(top=0.88)
# plt.show()
fig.savefig("MeasurementI.pdf", format="pdf")

# http://www.physics.emory.edu/faculty/weeks/idl/gofr.html


def pair_corr(parameters, t_therm, rxlog, rylog):
	L = parameters['L']
	phase = parameters['Phase']
	dt = parameters['dt']
	t_total = parameters['Nstep'] * dt
	N = parameters['Nx'] * parameters['Ny']

	dr = 0.01 * L
	t_range = np.arange(t_therm, t_total, dt)
	r_range = np.arange(0, L * np.sqrt(2), dr)
	hist_buck = []

	for t_index in range(len(t_range)):
		for ple_index in range(N):
			# calculate the distance between the current ple with other ple-s
			dx_temp = rxlog[:, t_index] - rxlog[ple_index, t_index]
			dy_temp = rylog[:, t_index] - rylog[ple_index, t_index]
			dx = np.delete(dx_temp, ple_index)
			dy = np.delete(dy_temp, ple_index)
			r = np.sqrt(dx ** 2 + dy ** 2)
			# https://stackoverflow.com/questions/9543935/python-count-occurrences-of-certain-ranges-in-a-list
			hist, bins = np.histogram(r, bins=r_range)
			hist = hist / (2.0 * np.pi * dr * N) / (N/L ** 2)
			hist = hist / bins[1:]
			hist_buck.append(hist)
	whole_hist = np.sum(np.array(hist_buck) / len(t_range), axis=0)

	return r_range[1:], whole_hist
	# Used to plot the interpolate lines, not much changes.
	# f2 = interpolate.interp1d(r_range[1:], whole_hist, kind="linear",fill_value="extrapolate")
	# new_r = np.arange(0, L * np.sqrt(2), 1e-1*dr)[1:]
	# new_hist = f2(new_r)
	# plt.plot(new_r, new_hist)


"""""""""
Measurement II
"""""""""
r_solid, hist_solid = pair_corr(solid_parameter, t_therm_solid, testa_x, testa_y)
r_fluid, hist_fluid = pair_corr(fluid_parameter, t_therm_fluid, testb_x, testb_y)

fig = plt.figure(figsize=(8,8))
fig.suptitle("Measurement II", fontsize=16)

axis1 = fig.add_subplot(211)
axis1.title.set_text('{0} g(r)'.format(solid_parameter["Phase"]))
axis1.plot(r_solid, hist_solid, 'k-', label="Line Plot")
# axis1.bar(r_solid, hist_solid, width=0.2, facecolor="None", label="Histogram")
axis1.legend()
axis1.set_xlabel("r")
axis1.set_ylabel("g(r)")

axis2 = fig.add_subplot(212)
axis2.title.set_text('{0} g(r)'.format(fluid_parameter["Phase"]))
axis2.plot(r_fluid, hist_fluid, 'k-', label="Line Plot")
# axis2.bar(r_fluid, hist_fluid, width=0.2, facecolor="None", label="Histogram")
axis2.legend()
axis2.set_xlabel("r")
axis2.set_ylabel("g(r)")

plt.tight_layout()
fig.subplots_adjust(top=0.88)
# plt.show()
fig.savefig("MeasurementII.pdf", format="pdf")
