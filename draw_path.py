# from path import PATH
import random
a = [[0,20],[20,0],[0,-20],[-20,0]]


import sys
import pygame
from pygame.locals import *
import numpy as np
b = np.zeros((20,30))

f = open("./position.txt", "r")
while True:
    string = f.readline()
    if len(string)==0: break
    tmp = string.split()
    b[int(tmp[0]), int(tmp[1])] = 1
f.close()
print b

path_counter = 0
# while True:
w = 400
h = 400
x = 20
y = 20
now = [210,210]
# path = PATH[path_counter]
path_counter +=1
screen = pygame.display.set_mode((w,h))
# pygame.draw.line(screen, (0, 200, 200), (0, 0), (600, 0), (1))
# pygame.draw.line(screen, (0, 200, 200), (0, 0), (0, 400), (1))
# pygame.draw.line(screen, (0, 200, 200), (0, 20), (600, 20), (1))
# pygame.draw.line(screen, (0, 200, 200), (20, 0), (20, 400), (1))

clock = pygame.time.Clock()

# while 1:
#     clock.tick(100)
#     pygame.display.update()
#     for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     sys.exit()
#                 elif event.type == KEYDOWN and event.key == K_ESCAPE:
#                     sys.exit()
#     x+=20
#     y+=20
#     if x >= h and y >= w:
#         break
#     pygame.draw.line(screen, (0, 200, 200), (0, y), (600, x), (1))
#     pygame.draw.line(screen, (0, 200, 200), (x, 0), (x, 400), (1))

for i in range(20):
    for j in range(20):
        if b[i][j] == 0:
            xx = i*20+10
            yy = j*20+10
            pygame.draw.circle(screen, (255, 255, 255), (yy, xx), 3, 3)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit()


# px = [6, 7, 8, 9, 10, 10, 10, 10, 11, 12, 13, 14, 15, 16, 15, 15, 15, 15, 16, 17, 17, 17, 17, 16, 15, 15, 15, 15, 15]
# py = [8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 13, 14]

# checkpoint = [[7,11],[15,8]]
# goal = [[10, 7],[5, 9],[17, 11], [15,14]]
# for pos in checkpoint:
#     pygame.draw.circle(screen, (255, 0, 0), (pos[1]*20+10, pos[0]*20+10), 5, 5)
# for pos in goal:
#     pygame.draw.circle(screen, (0, 255, 0), (pos[1]*20+10, pos[0]*20+10), 5, 5)

for event in pygame.event.get():
    if event.type == pygame.QUIT:
        sys.exit()
    elif event.type == KEYDOWN and event.key == K_ESCAPE:
        sys.exit()

for i in range (20):
    for j in range(20):
        for xx in range(2):
            for yy in range(2):
                if xx+yy != 0:
                    x = i+xx-1
                    y = j+yy-1
                    print b[i][j] * b[x][y]
                    print (i-x)*(i-x) + (j-y)*(j-y)
                    if b[i][j] + b[x][y] == 0 and (i-x)*(i-x) + (j-y)*(j-y) <= 1:
                        pygame.draw.line(screen, (0, 255, 0), (j*20+10, i*20+10), (y*20+10, x*20+10), (10))
pygame.display.update()
pygame.image.save(screen, "screenshot.jpeg")
    # counter = 0
    # previous = [0,0]
    # previous[0] = px[0]*20+10
    # previous[1] = py[0]*20+10
    # xx = px[counter]*20+10
    # yy = py[counter]*20+10
    # while counter < len(px):

    #     pygame.draw.line(screen, (0, 255, 0), (previous[1],previous[0]), (yy,xx), (10))
    #     clock.tick(2)
    #     pygame.display.update()
    #     xx = px[counter]*20+10
    #     yy = py[counter]*20+10
    #     counter += 1
    #     pygame.draw.line(screen, (255, 0, 0), (previous[1],previous[0]), (yy,xx), (10))
    #     previous[0] = xx
    #     previous[1] = yy

    #previous = [210, 210]
    # while 1:
    #     clock.tick(2)
    #     pygame.display.update()
    #     for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     sys.exit()
    #                 elif event.type == KEYDOWN and event.key == K_ESCAPE:
    #                     sys.exit()
    #     pygame.draw.line(screen, (0, 255, 0), previous, now, (10))
    #     previous[0] = now[0]
    #     previous[1] = now[1]
    #     tmp = random.randint(0, 3)
    #     print (a[tmp])
    #     now[0] += a[tmp][0]
    #     now[1] += a[tmp][1]
    #     print(previous,"--------" ,now)
    #     if now[0] < 10 or now[1] < 10:
    #         now[0] = previous[0]
    #         now[1] = previous[1]
    #     else:
    #         pygame.draw.line(screen, (255, 0, 0), previous, now, (10))

    # counter = 0
    # previous = [0,0]
    # previous[0] = path[0][0]*20+10
    # previous[1] = path[0][1]*20+10
    # while counter < len(path):

    #     clock.tick(5)
    #     pygame.display.update()
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             sys.exit()
    #         elif event.type == KEYDOWN and event.key == K_ESCAPE:
    #             sys.exit()
    #     xx = path[counter][0]*20+10
    #     yy = path[counter][1]*20+10
    #     counter += 1
    #     pygame.draw.line(screen, (255, 0, 0), (previous[1],previous[0]), (yy,xx), (10))
    #     previous[0] = xx
    #     previous[1] = yy
    #     if counter==len(path):
    #         clock.tick(1)


