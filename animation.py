""""""""""""""""""
""" IMPORTS """
""""""""""""""""""

from drone_home_model import run_sim, DroneHomeModel
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


""""""""""""""""""
""" RUN SIM """
""""""""""""""""""


sim_parameters = {
    'size': 500,
    'steps': 10000,
    'population' : 10,
    'detection_radius' : 50,
    'camera_radius' : 7,
}

# Run simulation and collect data for animation and analysis
model, results = run_sim(DroneHomeModel, sim_parameters)
data = results.variables.DroneHomeModel

""""""""""""""""""""""""
""" INIT FIGURE """
""""""""""""""""""""""""

# Make plot and intial line scatter
fig, ax = plt.subplots(figsize=(5,5))

ax.set_facecolor('b')

uavs, = ax.plot([], [], 'wo')
person, = ax.plot([], [], 'rX')
time_text = ax.text(0.05, 0.95,'')

""""""""""""""""""
""" ANIMATION """
""""""""""""""""""

def init():
    ax.set_xlim(0, model.p.size)
    ax.set_ylim(0, model.p.size)
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    return uavs, person, time_text

def update(frame):
    # Plot uavs
    uav_xpos = data['pos'][frame][0]
    uav_ypos = data['pos'][frame][1]
    uavs.set_data(uav_xpos, uav_ypos)
    # Plot person
    person_xpos = data['person'][frame][0]
    person_ypos = data['person'][frame][1]
    person.set_data(person_xpos, person_ypos)
    # Write time
    time_text.set(text=f"Timestep: {frame}")
    return uavs, person, time_text

def animate(save = False):
    ani = FuncAnimation(fig, update, frames=data.index,
                    init_func=init, blit=True, 
                    interval=20, repeat=True)
    if save:
        ani.save("RWWD.mp4", progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))

    plt.show()

if __name__ == "__main__":
    animate()