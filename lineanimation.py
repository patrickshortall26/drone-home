""""""""""""""""""
""" IMPORTS """
""""""""""""""""""

from drone_home_model import run_sim, DroneHomeModel
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
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

ax.set_facecolor('b')

uavs, = ax.plot([], [], 'w-', markersize=1)
person, = ax.plot([], [], 'rX')
time_text = ax.text(0.05, 0.95,'')
search_space = Rectangle((data["search_center"][0][0]-(model.p.size/6 + 5), data["search_center"][0][1]-(model.p.size/6 + 5)),model.p.size/3,model.p.size/3,
                    edgecolor='red',
                    facecolor='none',
                    lw=1)

""""""""""""""""""
""" ANIMATION """
""""""""""""""""""

def init():
    ax.set_xlim(0, model.p.size/2)
    ax.set_ylim(model.p.size/2, model.p.size)
    ax.add_patch(search_space)
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    return uavs, person, time_text, search_space

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
    # Change search space center
    search_space.set_xy((data["search_center"][frame][0]-(model.p.size/6), data["search_center"][frame][1]-(model.p.size/6)))
    return uavs, person, time_text, search_space

def animate(save = False):
    ani = FuncAnimation(fig, update, frames=data.index,
                    init_func=init, blit=True, 
                    interval=20, repeat=True)
    if save:
        ani.save("cool_lines.mp4", progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))

    plt.show()

if __name__ == "__main__":
    animate(True)