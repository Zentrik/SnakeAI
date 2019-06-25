import pygame
from pygame.locals import *
import numpy as np

class game:
    def __init__(self, w, h, sols, parents):
        self.width = w
        self.height = h
        self.display = pygame.display.set_mode((self.width, self.height))
        self.stop = 0
        
        self.input = np.zeros(6) #6 element list input layer, 1-4 is distance to collision (right,left,up,down), 4-5 is distance to apple(x,y)
        self.hiddenLayer = np.zeros(5) #5 element list hidden Layer
        self.output = np.zeros(4) # 4 element list node output layer, right, left, up, down
        
        self.solutionsPerPopulation = sols
        self.parents = parents

        self.reward = np.zeros(self.solutionsPerPopulation) #how many apples eaten and steps taken towards apple,
        #self.weight = np.random.rand(self.solutionsPerPopulation, len(self.hiddenLayer), len(self.input)) #a weight per hidden layer node for each input
        #self.weight2 = np.random.rand(self.solutionsPerPopulation, len(self.output), len(self.hiddenLayer))

        self.numberOfWeights = len(self.hiddenLayer) * (len(self.input) + len(self.output)) # per solution
        self.weights = np.random.uniform(-1,1, [self.solutionsPerPopulation, self.numberOfWeights])

        pygame.init()

    def play(self):
        while not self.stop:
            self.reward = np.zeros(self.solutionsPerPopulation) #reset reward every population
            for n in range(self.solutionsPerPopulation):
                self.snake_head = [2 * self.width/25, self.height/2]    
                self.snake_position = [[2 * self.width/25, self.height/2],[self.width/25,self.height/2],[0, self.height/2]] 
                #self.snake_head = [2 * self.width/25, 0]    
                #self.snake_position = [[2 * self.width/25, 0],[self.width/25,0],[0, 0]]
                self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25]
                self.move = np.array([[self.width/25], [-self.height/25]])
                self.moving = np.array([[1],[0]]) #y,x
                self.alive = 1

                while self.reward[n] > -500 and self.alive and not self.stop:
                    self.button_press()
                    self.update(n)

            if np.max(self.reward) > 10000:
                break

            parentsIndex = np.argpartition(self.reward, -self.parents)[-self.parents:] # get top 20 performing chromosomes, index of reward array

            parentsWeights = np.empty([self.parents, self.numberOfWeights]) #create temporary array storing parent weights
            for i in range(len(parentsIndex)):
                parentsWeights[i] = self.weights[parentsIndex[i]] #populate temporary array with values

            print(np.max(self.reward)) #print top fitness

            for solution in range(self.solutionsPerPopulation): #make desired number of solutions
                while True:
                    p1 = np.random.randint(self.parents) #random parent index to obtain elements from parentsWeights list
                    p2 = np.random.randint(self.parents)
                    # produce offspring from two parents if they are different
                    if p1 != p2: #if two different parents
                        for weight in range(self.numberOfWeights): #loop for every weight, loop n times where n is equal to number of weights per solution/parent
                            if np.random.random() < 0.5:
                                self.weights[solution, weight] = parentsWeights[p1, weight]
                            else:
                                self.weights[solution, weight] = parentsWeights[p2, weight]
                        break #when for loop finished, i.e. every weight has been assigned

            for i in range(self.solutionsPerPopulation): #loop n times, where n is number of solutions, i.e. mutate every solution/chromosome
                self.weights[i, np.random.randint(self.numberOfWeights)] += np.random.uniform(-1,1) # add random value to random weight for every solution

        np.save("bestSolutionGlobal.npy", self.weights[np.argpartition(self.reward, -1)[-1:][0]])
    
    def update(self, solution):
        if not self.alive: #if game quite by button press return and end while loop
            return 

        self.display.fill((200,200,200)) #reset display
        pygame.draw.rect(self.display, (0,0,255) ,(self.apple_position[0], self.apple_position[1], self.width/25, self.height/25)) #draw apple
        
        for position in self.snake_position: #draw snake
            pygame.draw.rect(self.display,(255,0, 0),(position[0],position[1],self.width/25, self.height/25))

        pygame.display.update() #update display 

        x = self.snake_position[0][0] == self.apple_position[0] #if snake top right is equal to apple top right
        y = self.snake_position[0][1] == self.apple_position[1] #if snake top lett is equal to apple top left

        if x and y : # is apple eaten
            self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25] #respawn apple
            self.reward[solution] += 1000
        else:
            self.snake_position.pop() #move snake  

        self.NeuralNet(solution) #get input from neural net

        self.snake_position.insert(0,list(self.snake_head)) # move snake

        if self.snake_head[0]>self.width or self.snake_head[0]<0 or self.snake_head[1]>self.height or self.snake_head[1]<0 or self.snake_position[0] in self.snake_position[1:]: #if collision
            self.alive = 0
            self.reward[solution] -= 150

    def NeuralNet(self, solution):
        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake

        self.input[4:6] = [distance[0] / self.width, distance[1] / self.height]

        #lateral distance of apple from snake divided by screen width, positive if apple to the right
        #vertical distance of apple from snake divided by screen height, positive is apple below

        #if snake_head[0] in self.snake_position[1:][0]: # if x is the same find vertical distance between snake parts
        #    self.input(1) = self.snake_head[0] - self.snake_position.index(snake_head[0])
        self.input[2] = self.snake_head[1] # top
        self.input[3] = self.height - self.snake_head[1] + self.width/25 #bottom
        self.input[1] = self.snake_head[0] #left
        self.input[0] = self.width - self.snake_head[0] #right

        for p in self.snake_position[1:]:
            if p[0] == self.snake_head[0]:
                a = self.snake_head[1] - p[1]
                if a > 0 and a < self.input[2]:
                    self.input[2] = a - self.width/25
                elif a < 0 and abs(a + self.width/25 ) < self.input[3]:
                    self.input[3] = abs(a + self.width/25)

            elif p[1] == self.snake_head[1]:   
                a = self.snake_head[0] - p[0]
                if a > 0 and a < self.input[1]:
                    self.input[1] = a - self.width/25
                elif a < 0 and abs(a + self.width/25 ) < self.input[0]:
                    self.input[0] = abs(a + self.width/25)
            
        self.input[2] = self.input[2] / self.height
        self.input[3] = self.input[3] / self.height
        self.input[1] = self.input[1] / self.width
        self.input[0] = self.input[0] / self.width

        for i in range(len(self.hiddenLayer)):
            self.hiddenLayer[i] = self.sigmoid(np.dot(self.input, self.weights[solution][len(self.input)*i:len(self.input)*(i+1)])) 
        for i in range(len(self.output)):
            self.output[i] = self.sigmoid(np.dot(self.hiddenLayer, self.weights[solution][len(self.input)*len(self.hiddenLayer)+i*len(self.hiddenLayer):len(self.input)*len(self.hiddenLayer)+(i+1)*len(self.hiddenLayer)])) 

        movements = {0: [[1],[0]], 1: [[-1],[0]], 2: [[0],[-1]], 3: [[0],[1]]}
        self.moving = movements[np.argmax(self.output)] #fix so that it checks for duplicate value
        
        m = self.move * self.moving
        self.snake_head[0] += m[0][0]
        self.snake_head[1] += m[1][0]

        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake  

        #occurs after first move
        if abs(distance[0] / self.width) < abs(self.input[4]) or abs(distance[1] / self.height) < abs(self.input[5]): #if got closer to apple globally
            self.reward[solution] += 1
        else:
            self.reward[solution] -= 1.5
    
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-1 * x)))

    def button_press(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop = 1

snake = game(500,500, 1000, 10)
snake.play()