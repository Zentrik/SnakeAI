import numpy as np
import sys

class game:
    def __init__(self, sols, parents,f):
        self.width = 25
        
        self.input = np.zeros(6) #6 element list input layer, 1-4 is distance to collision (right,left,up,down), 4-5 is distance to apple(x,y)
        self.hiddenLayer = np.zeros(5) #5 element list hidden Layer
        self.output = np.zeros(4) # 4 element list node output layer, right, left, up, down
        
        self.solutionsPerPopulation = sols
        self.parents = parents

        self.numberOfWeights = len(self.hiddenLayer) * (len(self.input) + len(self.output)) # per solution

        self.weights = np.random.uniform(-1,1, [self.solutionsPerPopulation, self.numberOfWeights])
        
        self.desiredFitness = f

    def play(self):
        while True:
            self.reward = np.zeros(self.solutionsPerPopulation) #reset reward every population #how many apples eaten and steps taken towards apple,
            for n in range(self.solutionsPerPopulation):
                for l in range(2):
                    self.snake_head = [2, self.width//2]    
                    self.snake_position = [[2, self.width//2],[1, self.width//2],[0, self.width//2]]  
                    self.apple_position = [np.random.randint(5,self.width), np.random.randint(self.width)] #make sure apple doesnt spawn on snake

                    self.moving = [1, 0] #x,y
                    self.alive = 1

                    while self.reward[n] > -100 and self.alive:
                        self.update(n)

            self.evaluate()
    
    def update(self, solution):
        if self.snake_head == self.apple_position: # if apple eaten
            while self.apple_position in self.snake_position: #make sure apple doesnt spawn on snake
                self.apple_position = [np.random.randint(self.width), np.random.randint(self.width)] 
            self.reward[solution] += 200
        else:
            self.snake_position.pop() #move snake 

        self.NeuralNet(solution) #get input from neural net
        self.snake_position.insert(0,list(self.snake_head)) # move snake

        if self.snake_head[0] >= self.width or self.snake_head[0] < 0 or self.snake_head[1] >= self.width or self.snake_head[1] < 0 or self.snake_position[0] in self.snake_position[1:]: #if collision
            self.alive = 0

    def NeuralNet(self, solution):
        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake

        self.input[4:6] = [distance[0] / self.width, distance[1] / self.width] #lateral distance of apple from snake divided by screen width, positive if apple to the right
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
            
        self.input[2] = self.input[2] / self.width
        self.input[3] = self.input[3] / self.width
        self.input[1] = self.input[1] / self.width
        self.input[0] = self.input[0] / self.width

        self.hiddenLayer = self.relu(np.matmul(self.weights[solution, :len(self.hiddenLayer) * len(self.input)].reshape([len(self.hiddenLayer), len(self.input)]), self.input)) # matrix multilication of weights matrix (a x number of weights) by input list ax1 matrix
        self.output = np.matmul(self.weights[solution][len(self.hiddenLayer) * len(self.input):].reshape([len(self.output), len(self.hiddenLayer)]), self.hiddenLayer)

        movements = {0: [1, 0], 1: [-1,0], 2: [0, -1], 3: [0, 1]}
        #backwards = {0: [-1, 0], 1: [1,0], 2: [0, 1], 3: [0, -1]} #if coming from this direction the snake would die
        
        #if self.moving != backwards[np.argmax(self.output)]:
        self.moving = movements[np.argmax(self.output)] #fix so that it checks for duplicate value
        
        self.snake_head[0] += self.moving[0]
        self.snake_head[1] += self.moving[1]

        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake  

        #occurs after first move
        if abs(distance[0] / self.width) < abs(self.input[4]) or abs(distance[1] / self.width) < abs(self.input[5]): #if got closer to apple globally
            self.reward[solution] += 1
        else:
            self.reward[solution] -= 3

    def relu(self, x):
        return x * (x > 0)

    def save(self):
        np.save("FinalSolution.npy", self.weights[np.argpartition(self.reward, -1)[-1:][0]])
        print("Finished!\n", self.weights[np.argpartition(self.reward, -1)[-1:][0]])

    def evaluate(self):
        if np.max(self.reward) > self.desiredFitness: #if desired fitness acheived: save weights; exit program
                self.save()
                sys.exit()

        np.save("bestSolution.npy", self.weights[np.argpartition(self.reward, -1)[-1:][0]])

        parentsIndex = np.argpartition(self.reward, -self.parents)[-self.parents:] # get top 20 performing chromosomes, index of reward array

        parentsWeights = self.weights[parentsIndex]

        print(np.max(self.reward)) #print top fitness

        for solution in range(self.solutionsPerPopulation): #make desired number of solutions
            while True:
                p1 = np.random.randint(self.parents) #random parent index to obtain elements from parentsWeights list
                p2 = np.random.randint(self.parents) 

                if p1 != p2: #if two different parents, produce offspring from two parents 
                    for weight in range(self.numberOfWeights): #loop for every weight, loop n times where n is equal to number of weights per solution/parent
                        if np.random.random() < 0.5:
                            self.weights[solution, weight] = parentsWeights[p1, weight]
                        else:
                            self.weights[solution, weight] = parentsWeights[p2, weight]
                    break #when for loop finished, i.e. every weight has been assigned

        for i in range(self.solutionsPerPopulation): #loop n times, where n is number of solutions, i.e. mutate every solution/chromosome
            self.weights[i, np.random.randint(self.numberOfWeights)] += np.random.uniform(-1,1) # add random value to random weight for every solution

snake = game(100, 20, 30000)
snake.play()