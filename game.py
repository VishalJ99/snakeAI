"""
snake game 
need
get state -> current state (what is this?)
play step -> reward, game_over, score (? why score, can be inferred from rewards?)


state = [danger straight, danger right, danger left, 
direction left, direction right, direction up, direction down,
food left, food right,  food up, food down]
"""

import numpy as np
import pygame
import random
from enum import Enum
from collections import namedtuple
random.seed(1)
pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20
import time

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
       # init game state
       self.direction = Direction.RIGHT
       self.directions_array = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
       
       
       self.head = Point(self.w/2, self.h/2)
       self.snake = [self.head, 
                     Point(self.head.x-BLOCK_SIZE, self.head.y),
                     Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
       
       self.score = 0
       self.food = None
       self._place_food()
       self.frame = 0
       
    
    def get_state(self):
     '''
     Returns current state binary array 

     elements of state_vector : [danger straight, danger right, danger left, 
     direction left, direction right, direction up, direction down,
     left, food right,  food up, food down]
     '''
     head_copy = self.head

     pt = self._move(head_copy,self.direction)
     danger_straight = int(self._is_collision(pt))
     
     pt = self._move(head_copy,np.roll(self.directions_array,-1)[0])
     danger_right = int(self._is_collision(pt))

     pt = self._move(head_copy, np.roll(self.directions_array,1)[0])
     danger_left = int(self._is_collision(pt))

     direction_up = 1 if self.direction == Direction.UP else 0
     direction_down = 1 if self.direction == Direction.DOWN else 0

     direction_left = 1 if self.direction == Direction.LEFT else 0
     direction_right = 1 if self.direction == Direction.RIGHT else 0
     
     x0,y0 = self.head.x, self.head.y
     x1,y1 = self.food.x, self.food.y

     xdiff = np.sign(x1-x0)
     ydiff = np.sign(y1-y0)


     food_up = 1 if ydiff<0 else 0
     food_down = 1 if ydiff>0 else 0
     food_left = 1 if xdiff<0 else 0
     food_right = 1 if xdiff>0 else 0

     state_vector = [danger_straight, danger_right, danger_left, direction_up, direction_down, direction_left, direction_right, food_up, food_down, food_left, food_right]

     return state_vector

     
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self,action_vector):
        '''
        Plays a single step and returns score, gameover and reward
        
        Params:
         action_vector : array
          binary array determing movement of snake in current step eg [straight,right,left] 
        '''
        pygame.event.pump()
        # find direction specified by action vector
        if action_vector[1]:
         self.directions_array = np.roll(self.directions_array,-1)
        elif action_vector[2]:
         self.directions_array = np.roll(self.directions_array,1)
        
        self.direction = self.directions_array[0]
        
        # 2. move
        self.head = self._move(self.head,self.direction) # update the head
        self.snake.insert(0, self.head)
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
            self.reward = 10
        else:
            self.reward = 0
            self.snake.pop()
        if self._is_collision(self.head) or self.frame > 100*len(self.snake):
            game_over = True
            self.reward = -10
            return game_over, self.score, self.reward
        # 5. update ui and clock
        self._update_ui()
        # time.sleep(5)
        self.clock.tick(SPEED)
        self.frame+=1
        
        # 6. return game over and score
        game_over = False
        return game_over, self.score, self.reward
    
    def _is_collision(self, point):
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, head, direction):
        x = head.x
        y = head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        return Point(x, y)
            