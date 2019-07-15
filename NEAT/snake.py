import os
import neat
import visualize
import numpy as np
import pickle

class game:
    def __init__(self):
        self.width = 50
        self.input = [0,0,0,0,0,0]
    
    def train(self, config_file):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)

        # Load Config
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to 300 generations.
        winner = p.run(self.play, 300)

        #save winnning genome
        with open("winner.pkl", "wb") as output:
            pickle.dump(winner, output, 1)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    def play(self, genomes, config):
        for _ , genome in genomes:
            genome.fitness = 0 
            net = neat.nn.FeedForwardNetwork.create(genome, config) 
            
            self.snake_head = [2, self.width//2]    
            self.snake_position = [[2, self.width//2],[1, self.width//2],[0, self.width//2]]  
            self.apple_position = [np.random.randint(5,self.width), np.random.randint(self.width)] #make sure apple doesnt spawn on snake

            self.moving = [1, 0] #y,x
            self.alive = 1

            while genome.fitness > -100 and self.alive:
                self.update(genome, net)

    def update(self, genome, net):
        if self.snake_head == self.apple_position: # if apple eaten
            while self.apple_position in self.snake_position: #make sure apple doesnt spawn on snake
                self.apple_position = [np.random.randint(self.width), np.random.randint(self.width)] 
            genome.fitness += 200
        else:
            self.snake_position.pop() #move snake 

        self.NeuralNet(genome, net) #get input from neural net
        self.snake_position.insert(0,list(self.snake_head)) # move snake

        if self.snake_head[0] >= self.width or self.snake_head[0] < 0 or self.snake_head[1] >= self.width or self.snake_head[1] < 0 or self.snake_position[0] in self.snake_position[1:]: #if collision
            self.alive = 0

    def NeuralNet(self, genome, net):
        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake

        self.input[4:6] = distance #lateral distance of apple from snake divided by screen width, positive if apple to the right
        #vertical distance of apple from snake divided by screen height, positive is apple below

        self.input[2] = self.snake_head[1] # top
        self.input[3] = self.width - 1  - self.snake_head[1] #bottom
        self.input[1] = self.snake_head[0] #left
        self.input[0] = self.width - 1  - self.snake_head[0] #right

        for p in self.snake_position[1:]:
            if p[0] == self.snake_head[0]:
                a = self.snake_head[1] - p[1]
                if a > 0 and a < self.input[2]:
                    self.input[2] = a - 1
                elif a < 0 and abs(a + 1 ) < self.input[3]:
                    self.input[3] = abs(a + 1)

            elif p[1] == self.snake_head[1]:   
                a = self.snake_head[0] - p[0]
                if a > 0 and a < self.input[1]:
                    self.input[1] = a - 1
                elif a < 0 and abs(a + 1 ) < self.input[0]:
                    self.input[0] = abs(a + 1)

        output = net.activate(self.input)

        movements = {0: [1, 0], 1: [-1,0], 2: [0, -1], 3: [0, 1]}
        self.moving = movements[np.argmax(output)] #fix so that it checks for duplicate value
        
        self.snake_head[0] += self.moving[0]
        self.snake_head[1] += self.moving[1]

        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake  

        #occurs after first move
        if abs(distance[0]) < abs(self.input[4]) or abs(distance[1]) < abs(self.input[5]): #if got closer to apple globally
            genome.fitness += 1
        else:
            genome.fitness -= 3

snake = game()
snake.train("config_feedforward")