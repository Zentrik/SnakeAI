import pygame
from pygame.locals import *
import numpy as np
import NeuralNet

class game:
    def __init__(self, w, h):
        self.width = w // 25 * 25
        self.height = h
        self.display = pygame.display.set_mode((self.width, self.height))
        self.stop = 0
        
        self.input = np.zeros(6) #6 element list input layer, 1-4 is distance to collision (right,left,up,down), 4-5 is distance to apple(x,y)
        self.hiddenLayer = np.zeros(5) #5 element list hidden Layer
        self.output = np.zeros(4) # 4 element list node output layer, right, left, up, down

        self.reward = 0 #how many apples eaten and steps taken towards apple,

        self.weights = np.load('bestSolutionheadless.npy')

        self.snake_head = [2 * self.width/25, 12* self.height/25]    
        self.snake_position = [[2 * self.width/25, 12* self.height/25],[self.width/25,12* self.height/25],[0, 12* self.height/25]]  
        self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25]
        while self.apple_position == self.snake_head: 
            self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25]
        
        self.moving = np.array([[1],[0]]) #y,x
        self.move = np.array([[self.width/25], [self.height/25]])
        
        self.alive = 1

        self.clock = pygame.time.Clock()
        pygame.init()

    def play(self):
        while self.reward > -500 and self.alive and not self.stop:
            self.update()
            self.button_press()
            self.clock.tick(5)
        print(self.reward)

    def update(self):
        if self.snake_head == self.apple_position: # if apple eaten
            self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25] #respawn apple
            self.reward += 1000
        else:
            self.snake_position.pop() #move snake 

        self.NeuralNet() #get input from neural net

        self.snake_position.insert(0,list(self.snake_head)) # move snake

        if self.snake_head[0] >= self.width or self.snake_head[0] < 0 or self.snake_head[1] >= self.height or self.snake_head[1] < 0 or self.snake_position[0] in self.snake_position[1:]: #if collision
            self.alive = 0
            return

        self.display.fill((200,200,200)) #reset display
        pygame.draw.rect(self.display, (0,0,255) ,(self.apple_position[0], self.apple_position[1], self.width/25, self.height/25)) #draw apple
        
        for position in self.snake_position: #draw snake
            pygame.draw.rect(self.display,(255,0, 0),(position[0],position[1],self.width/25, self.height/25))

        pygame.display.update() #update display 

    def NeuralNet(self):
        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake

        self.input[4:6] = [distance[0] / self.width, distance[1] / self.height] #lateral distance of apple from snake divided by screen width, positive if apple to the right
        #vertical distance of apple from snake divided by screen height, positive is apple below

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

        self.hiddenLayer = NeuralNet.leaky_relu(np.matmul(self.weights[:len(self.hiddenLayer) * len(self.input)].reshape([len(self.hiddenLayer), len(self.input)]), self.input)) # matrix multilication of weights matrix (a x number of weights) by input list ax1 matrix
        self.output = NeuralNet.leaky_relu(np.matmul(self.weights[len(self.hiddenLayer) * len(self.input):].reshape([len(self.output), len(self.hiddenLayer)]), self.hiddenLayer)) 

        movements = {0: [[1],[0]], 1: [[-1],[0]], 2: [[0],[-1]], 3: [[0],[1]]}
        self.moving = movements[np.argmax(self.output)] #fix so that it checks for duplicate value

        m = self.move * self.moving
        self.snake_head[0] += m[0][0]
        self.snake_head[1] += m[1][0]

        distance = self.apple_position[0] - self.snake_head[0], self.apple_position[1] - self.snake_head[1] # distance between apple and snake  

        #occurs after first move
        if abs(distance[0] / self.width) < abs(self.input[4]) or abs(distance[1] / self.height) < abs(self.input[5]): #if got closer to apple globally
            self.reward += 1
        else:
            self.reward -= 1.5

    def button_press(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop = 1

snake = game(500,500)
snake.play()