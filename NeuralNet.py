import numpy as np

def NeuralNet(self, solution):
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

    self.hiddenLayer = leaky_relu(np.matmul(self.weights[solution, :len(self.hiddenLayer) * len(self.input)].reshape([len(self.hiddenLayer), len(self.input)]), self.input)) # matrix multilication of weights matrix (a x number of weights) by input list ax1 matrix
    self.output = leaky_relu(np.matmul(self.weights[solution, len(self.hiddenLayer) * len(self.input):].reshape([len(self.output), len(self.hiddenLayer)]), self.hiddenLayer)) 

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

def leaky_relu(x):
    return np.maximum(0.1 * x, x)