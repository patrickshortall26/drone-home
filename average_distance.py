""""""""""""""""""
""" IMPORTS """
""""""""""""""""""

from drone_home_model import run_sim, DroneHomeModel
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt


""""""""""""""""""
""" RUN SIM """
""""""""""""""""""

sim_parameters = {
    'size': 300,
    'seed' : 2439,
    'steps': 10000,
    'population' : 5,
    'dispersion_radius' : 25,
    'camera_radius' : 5,
    'random_walk_strength' : 1,
    'random_walk_delay' : 100,
    'border_strength' : 0.5,
    'sborder_strength' : 0.5,
    'dispersion_strength' : 5,
    'drone_speed' : 0.3
}

# Run simulation and collect data for animation and analyesis
model, results = run_sim(DroneHomeModel, sim_parameters)
data = results.variables.DroneHomeModel

""""""""""""""""""""""""
""" INIT FIGURE """
""""""""""""""""""""""""

# Make plot and intial line scatter
fig, ax = plt.subplots(figsize=(5,5))
fig.patch.set_facecolor('#004AAD')

ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

timestep = range(len(data.index))
uav_diff = np.zeros(len(data.index))

# Loop through each timestep in the data
for t, posistions in enumerate(data['pos']):
    xdiff = np.zeros(5)
    # Loop through all the x values and fine the mean difference
    for i, positiion in enumerate(posistions[0]):
        diff = np.absolute(posistions[0]-positiion)
        mean_diff = np.mean(diff)
        xdiff[i] = mean_diff
    ydiff = np.zeros(5)
    # Loop through all the y values and fine the mean difference
    for i, positiion in enumerate(posistions[1]):
        diff = np.absolute(posistions[1]-positiion)
        mean_diff = np.mean(diff)
        ydiff[i] = mean_diff
    tot_diff = np.sqrt(np.mean(xdiff) + np.mean(ydiff))
    uav_diff[t] = tot_diff
    
line, = ax.plot([], [], color='w')

""""""""""""""""""
""" ANIMATION """
""""""""""""""""""

def init():
    ax.set_facecolor('b')
    ax.set(xlabel = 't', ylabel = 'Average distance between drones')
    ax.set_xlim(0, len(data.index))
    ax.set_ylim(0, 10)
    return line,

def update(frame, timestep, uav_diff, line, ax):
    line.set_data(timestep[:frame], uav_diff[:frame])
    #line.axes.axis([0, len(uav_diff[:frame])+10, np.max((0,uav_diff[frame]-2)), uav_diff[frame]+2])
    ax.set_xlim(0, len(uav_diff[:frame])+10)
    ax.set_ylim(np.max((0,uav_diff[frame]-2)), uav_diff[frame]+2)
    return line, ax

def animate(save = False):
    ani = FuncAnimation(fig, update, frames=data.index,
                    fargs=[timestep, uav_diff, line, ax],
                    init_func=init, blit=False, 
                    interval=20, repeat=True)
    if save:
        ani.save("avge_dist.mp4", progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))

    plt.show()

if __name__ == "__main__":
    animate(True)