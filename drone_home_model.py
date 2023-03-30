""""""""""""""""""
""" IMPORTS """
""""""""""""""""""

# Model design
import agentpy as ap
import numpy as np
from math import pi

# Visualization
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt

""""""""""""""""""""""""
""" GENERAL FUNCTIONS """
""""""""""""""""""""""""

def normalize(v):
    """ 
    Normalize a vector to length 1
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

""""""""""""""""""""""""
""" AGENT BEHAVIOUR """
""""""""""""""""""""""""

class DroneHomeDrone(ap.Agent):
    """ 
    The drones
    """

    """"""""""""
    """ INIT """
    """"""""""""

    def setup(self):
        """
        Initialise agents a random velocity
        """
        # Random velocity
        self.velocity = normalize(
            self.model.nprandom.random(2) - 0.5)
        

    def setup_pos(self, space):
        """
        Set up agent position in space
        """
        self.space = space
        self.neighbors = space.neighbors
        # Put them in the middle
        space.positions[self][:] = self.p.size/2
        self.pos = space.positions[self]
    
    """"""""""""""""""
    """ MOVEMENT """
    """"""""""""""""""

    def update_velocity(self):
        """
        Update velocity
        """

        # Velocity 1 - Random walk
        v1 = np.zeros(2)
        if self.model.counter == 5:
            v1 = normalize(self.model.nprandom.random(2)-0.5)
            self.model.counter = 0

        # Velocity 2 - Border avoidance
        v2 = np.zeros(2)
        for i in range(2):
            if self.pos[i] < 5:
                v2[i] += 0.5
            elif self.pos[i] > self.space.shape[i] - 5:
                v2[i] -= 0.5

        # Velocity 3 - Dispersion
        v3 = np.zeros(2)
        for nb in self.neighbors(self, distance=self.p.detection_radius):
            # Calculate displacement (for direction) and distance (for repulsive strength) between them
            displacement = (self.pos - nb.pos)
            distance = np.linalg.norm(displacement)
            # Head in the opposite direction with a strength depnding on the distance (if they're not on top of each other)
            if distance > 0:
                v3 += (10/distance)*normalize(displacement)

        # Add velocities to current velocity and normalise
        
        self.velocity = normalize(self.velocity + v1 + v2 + v3)


    def update_position(self):
        """
        Updates the position of an agent with the updated velocity
        computed above
        """
        self.space.move_by(self, self.velocity)

    def person_check(self):
        """
        Check surroundings for a person
        """
        # Calculate distance between them and person
        distance = np.linalg.norm(self.pos-self.model.person)
        # See if distance is within their camera radius
        if distance < self.p.camera_radius:
            # Stop simulation
            self.model.stop()


""""""""""""""""""""""""
""" THE WHOLE MODEL """
""""""""""""""""""""""""

class DroneHomeModel(ap.Model):
    """
    A simulation to show how drone-home drones will effectively search an area
    """

    def setup(self):
        """ 
        Initializes the agents and network of the model
        """
        # Initialise space and add agents to positions in space
        self.space = ap.Space(self, shape=[self.p.size]*2)
        self.agents = ap.AgentList(self, self.p.population, DroneHomeDrone)
        self.space.add_agents(self.agents, random=True)
        self.agents.setup_pos(self.space)
        # Setup a counter so that agents can change velocity randomly every ten seconds
        self.counter = 0
        # Set up person at a random point in the sea
        self.person = self.model.nprandom.random(2)*self.p.size

    def step(self):
        """ 
        What to do and in what order at each timestep
        """
        # Check if they can see a person
        self.agents.person_check()
        # Adjust direction
        self.agents.update_velocity()
        # Move into new direction
        self.agents.update_position()
        # Move the person along with the drift
        
        # Add one to the counter
        self.counter += 1
    

    def update(self):
        """
        Things to record and checks to make at each time step 
        """
        # Record agents positions
        pos = self.space.positions.values() # Get agent's positions
        pos = np.array(tuple(pos)).T
        self.record("pos", pos)
        # Record position of person
        self.record("person", tuple(self.person))

    def end(self):
        """
        Anything to record/report when t = final, i.e. at the end of the simulation
        """


""""""""""""""""""
""" Run model """ 
""""""""""""""""""

def run_sim(m, p):
    """ 
    Run model and collect results
    """
    # Run model and collect results 
    model = m(p)
    results = model.run()
    return model, results

def run_exp(model, parameters, runs, random=True):
    """
    Run experiment for a set of parameters
    """
    # Get 10 different random seeds 
    sample = ap.Sample(parameters, runs, randomize=random)
    # Run experiment and gather results
    exp = ap.Experiment(model, sample, record=True)
    results = exp.run(-1, display=False)
    return results