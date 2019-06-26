import numpy as np
import NeuralNet

class game:
    def __init__(self, w, h, sols, parents):
        self.width = w
        self.height = h
        self.move = np.array([[self.width/25], [self.height/25]])
        
        self.input = np.zeros(6) #6 element list input layer, 1-4 is distance to collision (right,left,up,down), 4-5 is distance to apple(x,y)
        self.hiddenLayer = np.zeros(5) #5 element list hidden Layer
        self.output = np.zeros(4) # 4 element list node output layer, right, left, up, down
        
        self.solutionsPerPopulation = sols
        self.parents = parents

        self.reward = np.zeros(self.solutionsPerPopulation) #how many apples eaten and steps taken towards apple,

        self.numberOfWeights = len(self.hiddenLayer) * (len(self.input) + len(self.output)) # per solution
        self.weights = np.random.uniform(-1,1, [self.solutionsPerPopulation, self.numberOfWeights])

    def play(self):
        while True:
            self.reward = np.zeros(self.solutionsPerPopulation) #reset reward every population
            for n in range(self.solutionsPerPopulation):
                self.snake_head = [2 * self.width/25, 12* self.height/25]    
                self.snake_position = [[2 * self.width/25, 12* self.height/25],[self.width/25,12* self.height/25],[0, 12* self.height/25]]

                self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25]
                while self.apple_position == self.snake_head: 
                    self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25]
                
                self.moving = np.array([[1],[0]]) #y,x
                self.alive = 1

                while self.reward[n] > -500 and self.alive:
                    self.update(n)

            if np.max(self.reward) > 100000:
                break

            parentsIndex = np.argpartition(self.reward, -self.parents)[-self.parents:] # get top 20 performing chromosomes, index of reward array

            parentsWeights = np.empty([self.parents, self.numberOfWeights]) #create temporary array storing parent weights
            for i in range(len(parentsIndex)):
                parentsWeights[i] = self.weights[parentsIndex[i]] #populate temporary array with values

            print(np.max(self.reward)) #print top fitness

            for solution in range(self.solutionsPerPopulation - self.parents): #make desired number of solutions
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

            for i in range(self.solutionsPerPopulation - self.parents): #loop n times, where n is number of solutions, i.e. mutate every solution/chromosome
                self.weights[i, np.random.randint(self.numberOfWeights)] += np.random.uniform(-1,1) # add random value to random weight for every solution
        
            self.weights[-self.parents:] = parentsWeights

        np.save("bestSolutionheadless.npy", self.weights[np.argpartition(self.reward, -1)[-1:][0]])
        print("Complete!")
    
    def update(self, solution):
        if self.snake_head == self.apple_position: # if apple eaten
            self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25] #respawn apple
            self.reward[solution] += 1000
        else:
            self.snake_position.pop() #move snake 

        NeuralNet.NeuralNet(self, solution) #get input from neural net

        self.snake_position.insert(0,list(self.snake_head)) # move snake

        if self.snake_head[0] >= self.width or self.snake_head[0] < 0 or self.snake_head[1] >= self.height or self.snake_head[1] < 0 or self.snake_position[0] in self.snake_position[1:]: #if collision
            self.alive = 0
            self.reward[solution] -= 200

snake = game(500,500, 100, 10)
snake.play()