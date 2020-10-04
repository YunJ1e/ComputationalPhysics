import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = "C:\FFmpeg\\bin\\ffmpeg.exe"

# The animation plotting program is built based on the example below
# https://matplotlib.org/3.3.1/api/animation_api.html
"""
	!!!README!!!
For solid, please uncomment line 17 to line 31 and line 54 to line 61
and at the same time comment the line 35 to 49 and line 65 to 72.

And for fluid, just do above reversely.

Please do not draw the animation for solid and fluid at the same time, it might work fine for 
solid, but it will draw the extra "solid particles" at the fluid canvas.

I will try to implement next time in a class or something. That might be the solution
"""
solid_data = np.loadtxt("positions_Verlet_Solid_6.dat")
fluid_data = np.loadtxt("positions_Verlet_Fluid_12.dat")


# def init_solid():
# 	ax.set_xlim(0, 6)
# 	ax.set_ylim(0, 6)
#
# 	return ln,
#
#
# def update_solid(i):
# 	xdata, ydata = [], []
# 	for N in range(36):
# 		xdata.append(solid_data[i*600 + 60000*N, 0])
# 		ydata.append(solid_data[i*600 + 60000*N, 1])
# 	ln.set_data(xdata, ydata)
# 	ax.set_title("Trajectory of solid at {0}/ 60000 frames".format(i * 600))
# 	return ln, ax,


def init_fluid():
	ax1_fluid.set_xlim(0, 12)
	ax1_fluid.set_ylim(0, 12)

	return ln_fluid,


def update_fluid(i):
	xdata_fluid, ydata_fluid = [], []
	for N in range(36):
		xdata_fluid.append(fluid_data[i*600 + 60000*N, 0])
		ydata_fluid.append(fluid_data[i*600 + 60000*N, 1])
	ln_fluid.set_data(xdata_fluid, ydata_fluid)
	ax1_fluid.set_title("Trajectory of fluid at {0}/ 60000 frames".format(i * 600))
	return ln_fluid, ax1_fluid,


Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)
# xdata, ydata = [], []
# ax.set_title(label='Let us begin', fontdict={'fontsize': 12}, loc='center')
# ln, = plt.plot([], [], 'bo')
# plt.grid(True)
# solid_ani = FuncAnimation(fig, update_solid, frames=int(60000/600),init_func=init_solid)
# solid_ani.save('Verlet_Solid_6.mp4', writer=writer)


fig1 = plt.figure(figsize=(8,8))
ax1_fluid = fig1.add_subplot(111)
xdata_fluid, ydata_fluid = [], []
ax1_fluid.set_title(label='Let us begin', fontdict={'fontsize': 12}, loc='center')
ln_fluid, = plt.plot([], [], 'ro')
plt.grid(True)
fluid_ani = FuncAnimation(fig1, update_fluid, frames=int(60000/600),init_func=init_fluid)
fluid_ani.save('Verlet_Fluid_12.mp4', writer=writer)