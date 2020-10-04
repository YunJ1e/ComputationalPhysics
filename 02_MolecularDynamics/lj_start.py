import numpy as np
import matplotlib.pyplot as plt


def initialize_positions_and_velocities(rx, ry, vx, vy, Nx, Ny, L):
	dx = L / Nx
	dy = L / Ny
	np.random.seed(0)
	for i in range(Nx):
		for j in range(Ny):
			rx[i * Ny + j] = dx * (i + 0.5)
			ry[i * Ny + j] = dy * (j + 0.5)

			u = np.random.random()  # This is box muller
			v = np.random.random()
			vx[i * Ny + j] = np.sqrt(-2 * np.log(u)) * np.cos(2. * np.pi * v)
			vy[i * Ny + j] = np.sqrt(-2 * np.log(u)) * np.sin(2. * np.pi * v)
	# subtract net velocity to avoid global drift
	vxav = sum(vx) / vx.size
	vyav = sum(vy) / vx.size
	vx -= vxav
	vy -= vyav


def force(rsq):
	# Implement the force equation here. Note that the force is minus the derivative of the potential.
	# What is passed into the function is the square of the interparticle distance.
	# What is used on line 79 is dx times the result of this function
	# Set the constant to 1
	epsilon = 1
	sigma = 1
	sigmasq = sigma ** 2
	r = np.sqrt(rsq)

	return (24 * epsilon / r) * ((-2) * (sigmasq / rsq) ** 6 + (sigmasq / rsq) ** 3)


def potential(rsq):
	rsqinv = 1. / rsq
	r6inv = rsqinv * rsqinv * rsqinv
	return -4 * r6inv * (1 - r6inv)


def compute_kinetic_energy(vx, vy):
	return 0.5 * sum(vx * vx + vy * vy)


def compute_potential_energy(rx, ry, rcut, L):
	rcutsq = rcut * rcut
	rcutv = potential(rcutsq)  # shift the potential to avoid jump at rc
	Epot = 0.
	for i in range(rx.size):
		for j in range(i):
			dx = rx[i] - rx[j]
			dy = ry[i] - ry[j]
			# minimum image convention
			if (dx > L / 2.): dx = dx - L
			if (dx < -L / 2.): dx = dx + L
			if (dy > L / 2.): dy = dy - L
			if (dy < -L / 2.): dy = dy + L
			# print dx,dy
			# compute the distance
			rsq = dx * dx + dy * dy
			if (rsq < rcutsq):
				Epot += potential(rsq) - rcutv
	return Epot


def compute_forces(rx, ry, dV_drx, dV_dry, N, L, rcut):
	rcutsq = rcut * rcut
	for i in range(N):
		for j in range(i):
			dx = rx[i] - rx[j]
			dy = ry[i] - ry[j]
			# minimum image convention
			if (dx > L / 2.): dx = dx - L
			if (dx < -L / 2.): dx = dx + L
			if (dy > L / 2.): dy = dy - L
			if (dy < -L / 2.): dy = dy + L
			# compute the distance
			rsq = dx * dx + dy * dy
			# check if we are < the cutoff radius
			if (rsq < rcutsq):
				# here is the call of the force calculation
				dV_dr = -force(rsq)

				# here the force is being added to the particle. Note the additional dx
				dV_drx[i] += dx * dV_dr
				dV_drx[j] -= dx * dV_dr
				dV_dry[i] += dy * dV_dr
				dV_dry[j] -= dy * dV_dr


def euler(rx, ry, vx, vy, dV_drx, dV_dry):
	deltat = 1e-3
	# update the positions
	rx += deltat * vx
	ry += deltat * vy

	# update the velocities
	vx -= deltat * dV_drx
	vy -= deltat * dV_dry


def verlet(rx, ry, vx, vy, dV_drx, dV_dry, N, L, rcut, dt):

	rx += dt * vx + 0.5 * dt ** 2 * (dV_drx)
	ry += dt * vy + 0.5 * dt ** 2 * (dV_dry)

	rebox(rx, ry, L)

	dV_drx_next = np.zeros(N)
	dV_dry_next = np.zeros(N)

	compute_forces(rx, ry, dV_drx, dV_dry, N, L, rcut)

	vx += 0.5 * dt * (dV_drx + dV_drx_next)
	vy += 0.5 * dt * (dV_dry + dV_dry_next)


# put back into box
def rebox(rx, ry, L):
	for i in range(rx.size):
		if rx[i] > L:
			rx[i] = rx[i] - L
		if rx[i] < 0:
			rx[i] = rx[i] + L
		if ry[i] > L:
			ry[i] = ry[i] - L
		if ry[i] < 0:
			ry[i] = ry[i] + L


def print_result(rxlog, rylog, vxlog, vylog, para_dict):
	fr = open("positions_{0}_{1}_{2}.dat".format(para_dict["Method"], para_dict["Phase"], para_dict["L"]), 'w')
	fv = open("velocities_{0}_{1}_{2}.dat".format(para_dict["Method"], para_dict["Phase"], para_dict["L"]), 'w')

	for j in range(rxlog.shape[1]):
		for i in range(rxlog.shape[0]):
			fr.write(str(rxlog[i, j]) + " " + str(rylog[i, j]) + '\n')
			fv.write(str(vxlog[i, j]) + " " + str(vylog[i, j]) + '\n')
	fr.write('\n')
	fv.write('\n')


def main(para_dict):
	# simulation parameters
	# set particles onto a grid initially
	Nx = para_dict["Nx"]
	Ny = para_dict["Ny"]
	N = Nx * Ny
	L = para_dict["L"]
	Nstep = para_dict["Nstep"]
	dt = para_dict["dt"]
	rcut = 2.5  # a usual choice for the cutoff radius

	vx = np.zeros(N)
	vy = np.zeros(N)
	rx = np.zeros(N)
	ry = np.zeros(N)

	rxlog = np.zeros([Nstep, N])
	rylog = np.zeros([Nstep, N])
	vxlog = np.zeros([Nstep, N])
	vylog = np.zeros([Nstep, N])

	initialize_positions_and_velocities(rx, ry, vx, vy, Nx, Ny, L)
	Etot = np.zeros(Nstep)

	for i in range(Nstep):
		dV_drx = np.zeros(N)
		dV_dry = np.zeros(N)
		compute_forces(rx, ry, dV_drx, dV_dry, N, L, rcut)

		# propagate using forward Euler
		if para_dict["Method"] == "Euler":
			euler(rx, ry, vx, vy, dV_drx, dV_dry)
		else:
			# Replace the FW Euler with a velocity Verlet
			verlet(rx, ry, vx, vy, dV_drx, dV_dry, N, L, rcut, dt)

		# make sure we're still in the box
		rebox(rx, ry, L)

		# keep track for printing
		rxlog[i] = rx
		rylog[i] = ry
		vxlog[i] = vx
		vylog[i] = vy

		# get some observables
		Epot = compute_potential_energy(rx, ry, rcut, L)
		Ekin = compute_kinetic_energy(vx, vy)
		Etot[i] = (Epot + Ekin)

	# Save the data into files
	# print_result(rxlog, rylog, vxlog, vylog, para_dict)

	return dt * np.arange(Nstep), Etot


def main_prob1():
	method_candidate = ["Euler", "Verlet"]
	# method_candidate = ["Verlet"]
	t = []
	E = []
	for method in method_candidate:
		solid_parameter = {"Nx": 6, "Ny": 6, "L": 6, "Nstep": int(6e4), "dt": 1e-3, "Method": method, "Phase": "Solid"}
		fluid_parameter = {"Nx": 6, "Ny": 6, "L": 12, "Nstep": int(6e4), "dt": 1e-3, "Method": method, "Phase": "Fluid"}
		print("L = {1} {0} Starts".format(method, solid_parameter["L"]))
		t_solid, E_solid = main(solid_parameter)
		print("L = {1} {0} Done".format(method, solid_parameter["L"]))
		t.append(t_solid)
		E.append(E_solid)

		print("Fluid {0} Starts".format(method))
		t_fluid, E_fluid = main(fluid_parameter)
		print("Fluid {0} Done".format(method))
		t.append(t_fluid)
		E.append(E_fluid)

	fig = plt.figure(figsize=(8, 8))
	fig.suptitle("Energy Visualization", fontsize=16)

	axis1 = fig.add_subplot(221)
	axis1.title.set_text("Solid Energy Method: Euler")
	axis1.plot(t[0], E[0])
	axis1.set_xlabel("Time(s)")
	axis1.set_ylabel("Energy Value")

	axis2 = fig.add_subplot(222)
	axis2.title.set_text("Fluid Energy Method: Euler")
	axis2.plot(t[1], E[1])
	axis2.set_xlabel("Time(s)")
	axis2.set_ylabel("Energy Value")

	axis3 = fig.add_subplot(223)
	axis3.title.set_text("Solid Energy Method: Verlet")
	axis3.plot(t[2], E[2])
	axis3.set_xlabel("Time(s)")
	axis3.set_ylabel("Energy Value")

	axis4 = fig.add_subplot(224)
	axis4.title.set_text("Fluid Energy Method: Verlet")
	axis4.plot(t[3], E[3])
	axis4.set_xlabel("Time(s)")
	axis4.set_ylabel("Energy Value")

	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	# plt.show()
	fig.savefig("Energy Visualization.pdf", format="pdf")


main_prob1()