import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys

'''Possible simulation regimes'''
SEXUAL_REPRO = True
HORIZ_GENET = False
DANGER_MAKES_NOISE  = False
MORTALITY = False

'''Amounts of energy conferred by the two kinds of light'''
RED_LIGHT_ENERGY = 5000.0
BLUE_LIGHT_ENERGY = 5000.0

'''Dimension of the arena'''
ARENA_XMIN = -1000
ARENA_XMAX = 1000
ARENA_YMIN = -1000
ARENA_YMAX = 1000

'''Greatest possible distance between points in arena'''
MAX_SQDIST = (ARENA_XMAX - ARENA_XMIN)**2 + (ARENA_YMAX - ARENA_YMIN)**2

'''Number of robots, number of neurons per robots'''
NUM_ROBOTS = 60
NEURONS_PER_ROBOT = 10

'''Physical parameters for robots'''
DIAMETER = 10.0
SPEED = 25.0

'''Eye angles'''
LEFT_EYE_MIN = 3*np.pi/12
LEFT_EYE_MAX = 5*np.pi/12
RIGHT_EYE_MIN = -3*np.pi/12
RIGHT_EYE_MAX = -5*np.pi/12

START_ENERGY = 60
MAX_ENERGY = 110  # Robots die if energy < 0 or > max_energy

LOW_ENERGY = 10   # If DANGER_MAKES_NOISE is true and energy amount is < low or > high, add danger noise to sensors
HIGH_ENERGY = 100

PERCEPTION_NOISE = 0.00001
PERCEPTION_GAIN = 50000

DANGER_NOISE = 0.0001

'''Genetic parameters'''
GENOME_LENGTH = NEURONS_PER_ROBOT + NEURONS_PER_ROBOT * NEURONS_PER_ROBOT # Biases and connection weights

MUTATION_PROB = 0.3     # Probability of a gene being mutated
MUTATION_SIGMA = 0.1    # Standard deviation of gene mutations (genes are floats in the range 0-1)


MAX_AGE_GENETRANSFER = 200  # Robots perform gene transfer if they are in danger and young enough (assunming that HORIZ_GENET is true)
PROB_HORIZGT = 0.1          # Probability (per second) that a robot that could perform gene transfer does in deed do so.
PROB_GENETR = 0.1           # Probability that a robot undergoing lateral gene transfer imports a gene from another robot

MAX_AGE = 500               # If MORTALITY is true, all robots past this age die.

'''Limits for neuron parameters'''
BIAS_MIN = -3
BIAS_MAX = 3
CONNECTION_MIN = -8
CONNECTION_MAX = 8

'''Neurons for perception and movement'''
NEURON_LEFTEYE_R = 0
NEURON_LEFTEYE_B = 1
NEURON_RIGHTEYE_R = 2
NEURON_RIGHTEYE_B = 3
NEURON_LEFTMOTOR = 4
NEURON_RIGHTMOTOR = 5


'''Parameters for lights'''
LIGHTS_INTENSITY = 20.0
NUM_REDLIGHTS = 1
NUM_BLUELIGHTS = 1

''' Durations of red and blue lights'''
REDLIGHTS_DMIN = 100
REDLIGHTS_DMAX = 200
BLUELIGHTS_DMIN = 100
BLUELIGHTS_DMAX = 200

''' Just to prevent the animations to be too slow and big, just take a frame every ten '''
ANIMATION_SPEEDUP = 10


class RotiManager:
    
    
    def __init__(self):
        ''' Initialize the whole scenario'''
        self.init_robots()
        self.init_lights()
        self.update_distances()
        
    def init_robots(self):
        '''Initialize the robots at the beginning of the simulation'''
        self.positions_x = np.zeros(NUM_ROBOTS)
        self.positions_y = np.zeros(NUM_ROBOTS)
        self.angles = np.zeros(NUM_ROBOTS)
        self.reyes = np.zeros(NUM_ROBOTS)
        self.leyes = np.zeros(NUM_ROBOTS)

        self.lefteyes_dx = np.zeros(NUM_ROBOTS)
        self.lefteyes_dy = np.zeros(NUM_ROBOTS)
        self.activations = np.zeros((NUM_ROBOTS, NEURONS_PER_ROBOT))
        self.firingrates = np.zeros((NUM_ROBOTS, NEURONS_PER_ROBOT))
        self.energy = np.zeros(NUM_ROBOTS)
        self.ages = np.zeros(NUM_ROBOTS)
        self.biasdevs = np.zeros((NUM_ROBOTS, NEURONS_PER_ROBOT))

        
        self.set_robots(np.ones(NUM_ROBOTS, dtype='Bool'))

        self.biases = np.zeros((NUM_ROBOTS, NEURONS_PER_ROBOT))
        self.connections = np.zeros((NUM_ROBOTS, NEURONS_PER_ROBOT, NEURONS_PER_ROBOT))
        
        self.genomes = np.random.rand(NUM_ROBOTS, GENOME_LENGTH) 
        self.set_neural_phenotypes(np.ones(NUM_ROBOTS, dtype='Bool'))

    def init_lights(self):
        '''Initialize the lights at the beginning of the scenario'''
        self.red_timeleft = np.zeros(NUM_REDLIGHTS)
        self.redx = np.zeros(NUM_REDLIGHTS)
        self.redy = np.zeros(NUM_REDLIGHTS)
        
        self.blue_timeleft = np.zeros(NUM_BLUELIGHTS)
        self.bluex = np.zeros(NUM_BLUELIGHTS)
        self.bluey = np.zeros(NUM_BLUELIGHTS)
        
        self.update_lights(1)

    def update_lights(self, dt):
        '''Update the light timecounts, move them if necessary, update the distances between robots and lights'''
        self.red_timeleft -= dt
        self.blue_timeleft -= dt
        
        red_toupdate = self.red_timeleft < 0
        howmany_red = np.sum(red_toupdate)
        
        self.redx[red_toupdate] = random_between(howmany_red, ARENA_XMIN, ARENA_XMAX)
        self.redy[red_toupdate] = random_between(howmany_red, ARENA_YMIN, ARENA_YMAX)
        self.red_timeleft[red_toupdate] = random_between(howmany_red, REDLIGHTS_DMIN, REDLIGHTS_DMAX)
        
        blue_toupdate = self.blue_timeleft < 0
        howmany_blue = np.sum(blue_toupdate)
        
        self.bluex[blue_toupdate] = random_between(howmany_blue, ARENA_XMIN, ARENA_XMAX)
        self.bluey[blue_toupdate] = random_between(howmany_blue, ARENA_YMIN, ARENA_YMAX)
        self.blue_timeleft[blue_toupdate] = random_between(howmany_blue, BLUELIGHTS_DMIN, BLUELIGHTS_DMAX)
        
        self.update_distances()
        
        
    def set_robots(self, to_set):
        ''' Set up the physical parameters of the robots (at the beginning of the simulation or after births)'''
        how_many = np.sum(to_set)
        self.positions_x[to_set] = random_between(how_many, ARENA_XMIN, ARENA_XMAX)
        self.positions_y[to_set] = random_between(how_many, ARENA_YMIN, ARENA_YMAX)
        self.angles[to_set] = random_between(how_many, 0, 2 * np.pi)
        self.reyes[to_set] = random_between(how_many, RIGHT_EYE_MIN, RIGHT_EYE_MAX)
        self.leyes[to_set] = random_between(how_many, LEFT_EYE_MIN, LEFT_EYE_MAX)
        
        
        self.activations[to_set, :] = np.zeros((how_many, NEURONS_PER_ROBOT))
        self.firingrates[to_set, :] = np.zeros((how_many, NEURONS_PER_ROBOT))
        self.energy[to_set] = np.ones(how_many) * START_ENERGY
        
        self.ages[to_set] = np.zeros(how_many)
        self.biasdevs[to_set] = np.zeros((how_many, NEURONS_PER_ROBOT))
    
    def set_neural_phenotypes(self, to_set):
        ''' Set up the neural phenotypes of the robots, on the basis of their genotypes'''
        how_many = np.sum(to_set)
        self.biases[to_set] = self.genomes[to_set, :NEURONS_PER_ROBOT] * (BIAS_MAX - BIAS_MIN) + BIAS_MIN
        self.connections[to_set] = np.reshape(self.genomes[to_set, NEURONS_PER_ROBOT:], (how_many, NEURONS_PER_ROBOT, NEURONS_PER_ROBOT)) * (CONNECTION_MAX - CONNECTION_MIN) + CONNECTION_MIN
        self.connections[to_set] *= np.expand_dims((1-np.eye(NEURONS_PER_ROBOT)),0)

        
    def update_distances(self):
        ''' Update the distances between robots and lights'''
        self.red_sqdists = np.sum((np.expand_dims(self.positions_x,1) - np.expand_dims(self.redx,0))**2 + 
                                  (np.expand_dims(self.positions_y,1) - np.expand_dims(self.redy,0))**2, 1)
        
        self.blue_sqdists = np.sum((np.expand_dims(self.positions_x,1) - np.expand_dims(self.bluex,0))**2 + 
                                  (np.expand_dims(self.positions_y,1) - np.expand_dims(self.bluey,0))**2, 1)
    
    def move_and_rotate(self, vleft, vright, dt):
        '''Move the robots according to the impulses to their motors'''
        totalv = vleft + vright # translational velocity
        rotationv = np.arctan2(vright-vleft, DIAMETER)  # rotational velocity
        
        trmov = totalv*dt
        romov = rotationv*dt
            
        self.angles += romov
        self.positions_x += trmov * np.cos(self.angles)
        self.positions_y += trmov * np.sin(self.angles)
        
        self.positions_x[self.positions_x < ARENA_XMIN] = ARENA_XMIN
        self.positions_x[self.positions_x > ARENA_XMAX] = ARENA_XMAX
        self.positions_y[self.positions_y < ARENA_YMIN] = ARENA_YMIN
        self.positions_y[self.positions_y > ARENA_YMAX] = ARENA_YMAX
        
            
        
    def sensor_input(self):
        ''' Compute the visual stimuli for all robots'''
        self.righteyes_dx = DIAMETER/2 * np.cos(self.angles  + self.reyes )
        self.righteyes_dy = DIAMETER/2 * np.sin(self.angles  + self.reyes )
        self.lefteyes_dx  = DIAMETER/2 * np.cos(self.angles  + self.leyes )
        self.lefteyes_dy  = DIAMETER/2 * np.sin(self.angles  + self.leyes )
        
        inputs_left_red = self.input_to_eyes(self.lefteyes_dx, self.lefteyes_dy, self.redx, self.redy)
        inputs_left_blue = self.input_to_eyes(self.lefteyes_dx, self.lefteyes_dy, self.bluex, self.bluey)

        inputs_right_red = self.input_to_eyes(self.righteyes_dx, self.righteyes_dy, self.redx, self.redy) 
        inputs_right_blue = self.input_to_eyes(self.righteyes_dx, self.righteyes_dy, self.bluex, self.bluey) 


        return [inputs_left_red, inputs_left_blue, inputs_right_red, inputs_right_blue]
    

    def input_to_eyes(self, eyes_dx, eyes_dy, lights_x, lights_y):
        ''' Compute the visual stimuli for all eyes'''
        perceptions = np.zeros(NUM_ROBOTS)

        rays_dx = np.expand_dims(self.positions_x + eyes_dx,1) - np.expand_dims(lights_x, 0)
        rays_dy = np.expand_dims(self.positions_y + eyes_dy,1) - np.expand_dims(lights_y, 0)
        
        
        visible_lights = (np.expand_dims(eyes_dx,1) * rays_dx) + (np.expand_dims(eyes_dy,1) * rays_dy) < 0
        any_visible = np.any(visible_lights, axis= 1)
                  
              
        perceptions[any_visible] = LIGHTS_INTENSITY/ (np.sum((rays_dx[any_visible] * visible_lights[any_visible])**2, 1) 
                                                      + np.sum((rays_dy[any_visible] * visible_lights[any_visible])**2, 1))
        
        noise =  np.random.normal(size=NUM_ROBOTS) * PERCEPTION_NOISE
        
        if(DANGER_MAKES_NOISE):
            indanger = np.logical_or(self.energy < LOW_ENERGY, self.energy > HIGH_ENERGY)
            howmany_danger = np.sum(indanger)
            noise[indanger] += np.random.normal(size = howmany_danger)* DANGER_NOISE
        
        return PERCEPTION_GAIN * (perceptions + noise)
    
    def perceive_and_move(self, dt):
        '''Main cycle: robots perceive, update their neural network, move, and then we update the lights, the energy, the ages of the robots, and we make robots die or be born if required'''
        perceptions = self.sensor_input()
        
        self.update_network(perceptions, dt)
        
        vleft = SPEED * self.firingrates[:, NEURON_LEFTMOTOR]
        vright = SPEED * self.firingrates[:, NEURON_RIGHTMOTOR]
        self.move_and_rotate(vleft, vright, dt)
    
        self.update_lights(dt)    
        self.update_energy(dt)
        self.age_robots(dt)
        self.death_and_birth()
    
    def update_energy(self, dt):
        '''Update the amount of energy available to each robot, on the basis on their distances from the lights and on the time elapsed since last update'''
        self.update_distances()
        self.energy += (RED_LIGHT_ENERGY/self.red_sqdists + BLUE_LIGHT_ENERGY/self.blue_sqdists) - dt
#        self.energy[self.energy > MAX_ENERGY] = MAX_ENERGY
            
        if HORIZ_GENET:
            self.horiz_gt(dt)
            
    def age_robots(self, dt):
        '''Just increase the ages of all robots'''
        self.ages += dt
        
    def death_and_birth(self):
        '''Kill all robots that have to die, allow an equal number of robots to be born'''
        if MORTALITY:
            dead = np.logical_or(np.logical_or(self.energy <0, self.ages > MAX_AGE), self.energy > MAX_ENERGY)
            #dead = np.logical_or(self.energy <0, self.ages > MAX_AGE)

        else:
            dead = np.logical_or(self.energy < 0, self.energy > MAX_ENERGY)
        
        #self.energy[self.energy>MAX_ENERGY] = MAX_ENERGY    
        
        if np.any(dead):
            self.newrobots(dead)

    
    def horiz_gt(self, dt):
        '''Perform horizontal gene transfer, if possible'''
        gt_possible = np.logical_and(np.logical_or(self.energy < LOW_ENERGY, self.energy > HIGH_ENERGY), self.ages < MAX_AGE_GENETRANSFER)
        howmany_possible = np.sum(gt_possible)
        do_horizgt = np.zeros(NUM_ROBOTS, dtype='Bool')
        do_horizgt[gt_possible] = np.random.rand(howmany_possible) < PROB_HORIZGT*dt
        howmany_gt = np.sum(do_horizgt)
        
        if (howmany_gt > 0):
            take_from = self.roulette_extract(howmany_gt)
            take_gene = np.random.rand(howmany_gt, GENOME_LENGTH) < PROB_GENETR
        
            
            self.genomes[do_horizgt, :][take_gene] = self.genomes[take_from, :][take_gene]
            self.mutate(self.genomes[do_horizgt])
            self.set_neural_phenotypes(do_horizgt)
        
    def newrobots(self, dead):
        '''Create new robots to replace the dead ones'''
        numnew = np.sum(dead)
        if (SEXUAL_REPRO): 
            parents1 = self.roulette_extract(numnew)
            parents2 = self.roulette_extract(numnew)
            newgenomes = self.crossover(parents1, parents2)
        else:
            parents = self.roulette_extract(numnew)
            newgenomes = np.copy(self.genomes[parents, :])
        
        newgenomes = self.mutate(newgenomes)
        
        self.genomes[dead,:] = newgenomes
        
        self.set_neural_phenotypes(dead)

        self.set_robots(dead)
    
    def roulette_extract(self, num_extractions):
        '''Extract robots (older ones more likely than younger ones) for sexual reproduction or horizontal gene transfer'''
        sumages = np.sum(self.ages)
        extracted = -np.ones(num_extractions, dtype='int64')
        
        values = np.random.rand(num_extractions) * sumages
        
        for i in range(NUM_ROBOTS): 
            values -= self.ages[i]
            extracted[np.logical_and(extracted < 0, values < 0)] = i 
        
        return extracted
    
    def crossover(self, parents1, parents2):
        '''Perform crossover for sexual reproduction'''
        numnew = len(parents1)
        genomesize = len(self.genomes[0])
        newgenomes = np.zeros((numnew, genomesize))
        
        fromparent1 = np.random.rand(numnew, genomesize) < 0.5
        fromparent2 = np.logical_not(fromparent1)
        
        newgenomes[fromparent1] = (self.genomes[parents1])[fromparent1]
        newgenomes[fromparent2] = (self.genomes[parents2])[fromparent2]
        
        
        
        return newgenomes
    
    def mutate(self, newgenomes):
        '''Mutate randomly all new genomes'''
        numnew = newgenomes.shape[0]
        genomesize = newgenomes.shape[1]
        
        mutate = np.random.rand(numnew)<MUTATION_PROB
        
        newgenomes[mutate] += np.random.normal(size=(sum(mutate), genomesize))*MUTATION_SIGMA
        newgenomes[newgenomes < 0] = 0
        newgenomes[newgenomes > 1] = 1
        
        return newgenomes
            
    def update_network(self, perceptions, dt):
        ''' Update the statuses of the neural networks of all robots'''
        d_act = - self.activations
        
        inputs_left_red = perceptions[0]
        inputs_left_blue = perceptions[1]
        inputs_right_red = perceptions[2]
        inputs_right_blue = perceptions[3]
        
        d_act[:, NEURON_LEFTEYE_R] += inputs_left_red
        d_act[:, NEURON_RIGHTEYE_R] += inputs_right_red
        d_act[:, NEURON_LEFTEYE_B] += inputs_left_blue
        d_act[:, NEURON_RIGHTEYE_B] += inputs_right_blue

        
        recurrent = np.einsum('ijk, ik -> ij', self.connections, self.firingrates)
        d_act += recurrent
        d_act *= dt
        self.activations += d_act
        self.firingrates = 1/(1 + np.exp(- (self.activations + self.biases + self.biasdevs)))

    def run(self, T, dt, createLog):
        '''Run the simulation for T seconds, with a step size of dt. If createLog is True, return a log containing the results of the simulation 
        (sped up according to the value of ANIMATION_SPEEDUP)'''
        nsteps = int(np.round(T/dt))
        nlogs = int(np.floor(nsteps/ANIMATION_SPEEDUP))
        if (createLog):
            log_x = np.zeros((nlogs, NUM_ROBOTS))
            log_y = np.zeros((nlogs, NUM_ROBOTS))
            log_angles = np.zeros((nlogs, NUM_ROBOTS))
            log_redx = np.zeros((nlogs, NUM_REDLIGHTS))
            log_redy = np.zeros((nlogs, NUM_REDLIGHTS))
            log_bluex = np.zeros((nlogs, NUM_BLUELIGHTS))
            log_bluey = np.zeros((nlogs, NUM_BLUELIGHTS))
            log_leyes = np.zeros((nlogs, NUM_ROBOTS))
            log_reyes = np.zeros((nlogs, NUM_ROBOTS))

        for i in range(nsteps):
            if (createLog) and i % ANIMATION_SPEEDUP == 0:
                j = i/ANIMATION_SPEEDUP
                log_x[j, :] = self.positions_x
                log_y[j, :] = self.positions_y
                log_angles[j, :] = self.angles
                log_redx[j, :] = self.redx
                log_redy[j, :] = self.redy
                log_bluex[j, :] = self.bluex
                log_bluey[j, :] = self.bluey
                log_leyes[j, :] = self.leyes
                log_reyes[j, :] = self.reyes
                
            self.perceive_and_move(dt)
            
            if (i % 2000 == 0):
                print("t = " + str(dt * i))
                sys.stdout.flush()
                
        if (createLog):
            return [log_x, log_y, log_angles, log_redx, log_redy, log_bluex, log_bluey, log_leyes, log_reyes]
    
    def visualize(self):
        fig = plt.figure()
        fig.patch.set_facecolor('white')
          
        plt.xlim(ARENA_XMIN, ARENA_XMAX)
        plt.ylim(ARENA_YMIN, ARENA_YMAX)       
        
        plt.plot(self.positions_x, self.positions_y, 'og', markersize=10)
        plt.plot([self.positions_x, self.positions_x + 35*np.cos(self.angles)], [self.positions_y, self.positions_y + 35*np.sin(self.angles)], 'k', linewidth=2)  
        plt.plot([self.positions_x, self.positions_x + 30*np.cos(self.angles + self.reyes)], [self.positions_y, self.positions_y + 30*np.sin(self.angles + self.reyes)], 'k', linewidth =2)  
        plt.plot([self.positions_x, self.positions_x + 30*np.cos(self.angles + self.leyes)], [self.positions_y, self.positions_y + 30*np.sin(self.angles + self.leyes)], 'k', linewidth = 2)  


        plt.plot(self.redx, self.redy, 'or', markersize=10)   
        plt.plot(self.bluex, self.bluey, 'ob', markersize=10)     
  
    def plot_histograms(self):
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.hist(self.ages)
        plt.title('Ages of robots')
        plt.ylabel('Number of robots')
        plt.xlabel('Age')
        
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.hist(self.energy)
        plt.title('Energy levels of robots')
        plt.ylabel('Number of robots')
        plt.xlabel('Energy')
        
        
        
        
def animate(log, title, save = False, filename='simulation.mp4'):
    
    log_x = log[0]
    log_y = log[1]
    log_angle = log[2]
    log_redx = log[3]
    log_redy = log[4]
    log_bluex = log[5]
    log_bluey = log[6]
    log_leyes = log[7]
    log_reyes = log[8]
    
    num_steps = log_x.shape[0]
    num_robots = log_x.shape[1]
        
    def animate_step(i, bodies, dirs, r_eyes, l_eyes, rlights, blights):
        if (i < num_steps):
            bodies.set_data(log_x[i, :], log_y[i,:])        
            for r in range(num_robots):
                dirs[r].set_data([log_x[i, r], log_x[i, r] + 25*np.cos(log_angle[i, r])], [log_y[i, r], log_y[i, r] + 25*np.sin(log_angle[i][r])])
                r_eyes[r].set_data([log_x[i, r], log_x[i, r] + 20*np.cos(log_angle[i, r] + log_reyes[i, r])], [log_y[i, r], log_y[i, r] + 20*np.sin(log_angle[i, r] + log_reyes[i, r])])
                l_eyes[r].set_data([log_x[i, r], log_x[i, r] + 20*np.cos(log_angle[i, r] + log_leyes[i, r])], [log_y[i, r], log_y[i, r] + 20*np.sin(log_angle[i, r] + log_leyes[i, r])])
                
        
            rlights.set_data(log_redx[i,:], log_redy[i,:])
            blights.set_data(log_bluex[i, :], log_bluey[i, :])
            return (bodies,dirs,r_eyes, l_eyes, rlights, blights)
    
    fig = plt.figure()
    plt.xlim(ARENA_XMIN, ARENA_XMAX)
    plt.ylim(ARENA_YMIN, ARENA_YMAX)
    plt.title(title)
    #plt.axes().set_aspect('equal', 'datalim')

    bodies, = plt.plot([], [], 'bo')
    
    dirs = np.empty(num_robots, dtype='object')
    l_eyes= np.empty(num_robots, dtype='object')
    r_eyes = np.empty(num_robots, dtype='object')


    for r in range(num_robots):
        dirs[r] = plt.plot([], [], 'g')[0]
        r_eyes[r] = plt.plot([], [], 'k')[0]
        l_eyes[r] = plt.plot([], [], 'k')[0]

    rlights,  = plt.plot([], [], 'or', markersize=10)
    blights,  = plt.plot([], [], 'ob', markersize=10)

    my_ani = anim.FuncAnimation(fig, animate_step, interval=2, save_count=num_steps, fargs = (bodies,dirs, r_eyes, l_eyes, rlights, blights))
    
    if save:
        my_ani.save(filename, fps = 20, bitrate=1900)
        print("Animation saved to " + filename)
    else:
        plt.show()

    
    
def random_between(n, MIN, MAX):
    return np.random.rand(n) * (MAX - MIN) + MIN