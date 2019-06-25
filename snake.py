import pygame
from pygame.locals import *
import numpy as np

class game:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.display = pygame.display.set_mode((self.width, self.height))
        self.stop = 0
        
        self.distance = []
        self.input = np.zeros(5) #4 element list input layer, 1 is is it clear foward, 2 is clear to the left, 3 clear to she right, 4 is apple forward/back, 5 is apple right/left
        self.hiddenLayer = np.zeros(5) #5 element list hidden Layer
        self.output = np.zeros(3) # 3 element list node output layer, forward, left, right
        
        self.solutionsPerPopulation = 100
        self.parents = 20

        self.reward = np.zeros(self.solutionsPerPopulation) #how many apples eaten and steps taken towards apple,
        #self.weight = np.random.rand(self.solutionsPerPopulation, len(self.hiddenLayer), len(self.input)) #a weight per hidden layer node for each input
        #self.weight2 = np.random.rand(self.solutionsPerPopulation, len(self.output), len(self.hiddenLayer))

        self.numberOfWeights = len(self.hiddenLayer) * (len(self.input) + len(self.output)) # per solution
        self.weights = np.random.rand(self.solutionsPerPopulation, self.numberOfWeights)

        pygame.init()

    def play(self):
        while not self.stop:
            self.reward = np.zeros(self.solutionsPerPopulation) #reset reward every population
            for n in range(self.solutionsPerPopulation):
                self.snake_head = [2 * self.width/25, 0]    
                self.snake_position = [[2 * self.width/25, 0],[self.width/25,0],[0, 0]] 
                self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25]
                self.move = np.array([[self.width/25], [-self.height/25]])
                self.moving = np.array([[1],[0]]) #y,x
                self.alive = 1

                while self.reward[n] > -250 and self.alive and not self.stop:
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

            '''weight3 = np.empty([self.parents, len(self.hiddenLayer), len(self.input)])
            weight4 = np.empty([self.parents, len(self.output), len(self.hiddenLayer)])

            for p in range(self.parents):

                weight3[p] = self.weight[parents[p]]
                weight4[p] = self.weight2[parents[p]]
            
            self.weight = weight3
            self.weight2 = weight4
            
            for n in range(0, 20, 2):
                p = np.array_split(self.weight[n],2)
                p2 = np.array_split(self.weight2[n],2)

                p3= np.array_split(self.weight[n+1],2)
                p4 = np.array_split(self.weight2[n+1],2)

                pp3 = np.append(p[0], p3[1], 0).reshape([1,len(self.hiddenLayer), len(self.input)]) 
                p3p = np.append(p3[0], p[1], 0).reshape([1,len(self.hiddenLayer), len(self.input)])
                p4p2 = np.append(p4[0], p2[1], 0).reshape([1,len(self.output), len(self.hiddenLayer)])
                p2p4 = np.append(p2[0], p4[1], 0).reshape([1,len(self.output), len(self.hiddenLayer)])     

                self.weight = np.append(self.weight, pp3, axis = 0)
                self.weight2 = np.append(self.weight2, p2p4, axis = 0)

                self.weight = np.append(self.weight, pp3, axis = 0)
                self.weight2 = np.append(self.weight2, p4p2, axis = 0)

                self.weight = np.append(self.weight, p3p, axis = 0)
                self.weight2 = np.append(self.weight2, p2p4, axis = 0)

                self.weight = np.append(self.weight, p3p, axis = 0)
                self.weight2 = np.append(self.weight2, p4p2, axis = 0)
        
            for n in range(self.solutionsPerPopulation - self.weight.shape[0]):
                r = np.random.randint(self.weight.shape[0])
                self.weight = np.append(self.weight, self.weight[r].reshape([1,len(self.hiddenLayer), len(self.input)]) , 0)
                self.weight2 = np.append(self.weight2, self.weight2[r].reshape([1,len(self.output), len(self.hiddenLayer)]), 0)
                self.weight[-1][np.random.randint(len(self.hiddenLayer))][np.random.randint(len(self.input))] = np.random.random()
                self.weight2[-1][np.random.randint(len(self.output))][np.random.randint(len(self.hiddenLayer))] = np.random.random()'''

        np.save("bestSolution.npy", self.weights[np.argpartition(self.reward, -1)[-1:][0]])
    
    def update(self, solution):
        if not self.alive: #if game quite by button press return and end while loop
            return 

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
            return

        self.display.fill((200,200,200)) #reset display
        pygame.draw.rect(self.display, (0,0,255) ,(self.apple_position[0], self.apple_position[1], self.width/25, self.height/25)) #draw apple
        
        for position in self.snake_position: #draw snake
            pygame.draw.rect(self.display,(255,0, 0),(position[0],position[1],self.width/25, self.height/25))

        pygame.display.update() #update display 

    def NeuralNet(self, solution):
        snake_head = np.array([[self.snake_head[0]],[self.snake_head[1]]])
        f = snake_head + self.move * self.moving # snake head after moving forward
        l = snake_head + self.move * (np.matmul(np.array([[0, -1],[1, 0]]), self.moving)) # snake head after moving leftwards
        r = snake_head + self.move * (np.matmul(np.array([[0, 1],[-1, 0]]), self.moving))

        
        apple_position = np.array([[self.apple_position[0]],[self.apple_position[1]]])
        distance = apple_position - snake_head # distance between apple and snake
        self.distance = [distance[0] / self.width, distance[1] / self.height]

        if np.array_equal(self.moving, np.array([[0],[-1]])): # if moving down
            distance = np.matmul(np.array([[-1, 0],[0, -1]]), distance)
        elif np.array_equal(self.moving, np.array([[1],[0]])): # if moving right
            distance = np.matmul(np.array([[0, 1],[-1, 0]]), distance)
        elif np.array_equal(self.moving, np.array([[-1],[0]])): # if moving left
            distance = np.matmul(np.array([[0, -1],[1, 0]]), distance)

        self.input[3] = distance[0] / self.width #lateral distance of apple from snake relative to snake direction and divided by screen width
        self.input[4] = -distance[1] / self.height #vertical distance of apple from snake relative to snake direction and divided by screen height, positive is forwards

        fnormal = [f[0][0], f[1][0]]
        lnormal = [l[0][0], l[1][0]]
        rnormal = [r[0][0], r[1][0]]

        if f[0][0]>self.width or f[0][0]<0 or f[1][0]>self.height or f[1][0]<0 or fnormal in self.snake_position[1:]:
            self.input[0] = 1
        if l[0][0]>self.width or l[0][0]<0 or l[1][0]>self.height or l[1][0]<0 or lnormal in self.snake_position[1:]:
            self.input[1] = 1
        if r[0][0]>self.width or r[0][0]<0 or r[1][0]>self.height or r[1][0]<0 or rnormal in self.snake_position[1:]:
            self.input[2] = 1

        for i in range(len(self.hiddenLayer)):
            self.hiddenLayer[i] = self.sigmoid(np.dot(self.input, self.weights[solution][len(self.input)*i:len(self.input)*(i+1)])) 
        for i in range(len(self.output)):
            self.output[i] = self.sigmoid(np.dot(self.hiddenLayer, self.weights[solution][len(self.input)*len(self.hiddenLayer)+i*len(self.hiddenLayer):len(self.input)*len(self.hiddenLayer)+(i+1)*len(self.hiddenLayer)])) 
        
        if abs(self.output[2]) > abs(self.output[1]):
            if abs(self.output[2]) > abs(self.output[0]):
                self.moving = np.matmul(np.array([[0, 1],[-1, 0]]), self.moving) # go right
        elif abs(self.output[1]) > abs(self.output[0]):
            self.moving = np.matmul(np.array([[0, -1],[1, 0]]), self.moving) # go left
        
        m = self.move * self.moving
        self.snake_head[0] += m[0][0]
        self.snake_head[1] += m[1][0]

        snake_head = np.array([[self.snake_head[0]],[self.snake_head[1]]])
        apple_position = np.array([[self.apple_position[0]],[self.apple_position[1]]])
        distance = apple_position - snake_head # distance between apple and snake
        #occurs after first move
        if abs(distance[0] / self.width) < abs(self.distance[0]) or abs(distance[1] / self.height) < abs(self.distance[1]): #if got closer to apple globally
            self.reward[solution] += 0.5
        else:
            self.reward[solution] -= 3
    
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-1 * x)))

    def button_press(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop = 1
            '''elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.move[0] != self.width/25:
                    self.move = [-self.width/25,0]
                elif event.key == pygame.K_RIGHT and self.move[0] != -self.width/25:
                    self.move = [self.width/25,0]
                elif event.key == pygame.K_UP and self.move[1] != self.height/25:
                    self.move = [0,-self.height/25]
                elif event.key == pygame.K_DOWN and self.move[1] != -self.height/25:
                    self.move = [0,self.height/25]'''

snake = game(500,500)
snake.play()