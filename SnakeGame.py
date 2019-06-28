import pygame
from pygame.locals import *
import numpy as np

class game:
    def __init__(self, w):
        self.width = w
        self.height = w
        self.display = pygame.display.set_mode((self.width, self.height))

        self.reward = 0 #how many apples eaten and steps taken towards apple,
        self.stop = 0

        self.clock = pygame.time.Clock()
        pygame.init()

    def play(self):
        while not self.stop:
            self.snake_head = [2 * self.width/25, 200]    
            self.snake_position = [[2 * self.width/25, 200],[self.width/25,200],[0, 200]] 
            self.apple_position = [np.random.randint(5, 25) * self.width/25, np.random.randint(25) * self.height/25]
            self.moving = [1,0] #y,x
            self.alive = 1
            self.reward = 0

            while self.alive:
                self.update()
                self.clock.tick(5)

            print(self.reward)
    
    def update(self):
        self.display.fill((200,200,200)) #reset display
        pygame.draw.rect(self.display, (0,0,255) ,(self.apple_position[0], self.apple_position[1], self.width/25, self.height/25)) #draw apple
        
        for position in self.snake_position: #draw snake
            pygame.draw.rect(self.display,(255,0, 0),(position[0],position[1],self.width/25, self.height/25))

        pygame.display.update() #update display 

        movements = {0: [1, 0], 1: [-1,0], 2: [0, -1], 3: [0, 1]}
        backwards = {0: [-1, 0], 1: [1,0], 2: [0, 1], 3: [0, -1]} #if coming from this direction the snake would die
        
        a = self.buttonpress()
        if a != 4 and self.moving != backwards[a]:
            self.moving = movements[a]

        self.snake_head[0] += self.moving[0] * self.width / 25
        self.snake_head[1] += self.moving[1] * self.width / 25
        
        self.snake_position.insert(0,list(self.snake_head)) # move snake

        if self.snake_head == self.apple_position: # if apple eaten
            self.apple_position = [np.random.randint(25) * self.width/25, np.random.randint(25) * self.height/25] #respawn apple
            self.reward += 1000
        else:
            self.snake_position.pop() #move snake 
        
        if self.snake_head[0] >= self.width or self.snake_head[0] < 0 or self.snake_head[1] >= self.height or self.snake_head[1] < 0 or self.snake_position[0] in self.snake_position[1:]: #if collision
            self.alive = 0

    def buttonpress(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop = 1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    return 1
                elif event.key == pygame.K_RIGHT:
                    return 0
                elif event.key == pygame.K_UP:
                    return 2
                elif event.key == pygame.K_DOWN:
                    return 3
        return 4

snake = game(500)
snake.play()