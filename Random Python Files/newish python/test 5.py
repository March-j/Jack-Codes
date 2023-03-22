import sys
import turtle
import random
import time

turtle.shape('classic')
turtle.color('blue', 'yellow')
turtle.speed(1000000)

width, height = turtle.window_width(), turtle.window_height()

minX, maxX = -width/2, width/2
minY, maxY = -height/2, height/2
a=random.randint(1,100)
b=random.randint(1,100)
c=random.randint(1,360)
d=random.randint(1,360)
5
h=1
g=-1
def repeated_code():
    global g
    global h
    global rad
    g=g+g-1
    turtle.circle(rad)
    turtle.up()
    turtle.setpos(0,-rad)
    turtle.down()
    

     # keep moving forward until we're out of window

    if not minX <= turtle.xcor() <= maxX or not minY <= turtle.ycor() <= maxY:
        turtle.penup()
        
        turtle.setpos(0,0)  # if we don't exit() a Terminator will be raised
        turtle.pendown()

    turtle.ontimer(repeated_code, 1)  # repeat every 1/10th of a second
for i in range(5):
    rad=20+20*i
    repeated_code()

    turtle.mainloop()
