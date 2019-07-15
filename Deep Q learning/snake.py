import numpy as np
import tensorflow as tf

class game:
    def __init__(self):
        self.width = 50
    
    def train(self, config_file):


    def play(self):
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

    def update(self):
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

    def NeuralNet(self):
        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake

        input = np.zeros([self.width,self.width])
        input(self.apple_position) = 1
        input(self.snake_head) = 2
        for i in self.snake_position[1:]:
            input(i) = 3
        self.input = input

        

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