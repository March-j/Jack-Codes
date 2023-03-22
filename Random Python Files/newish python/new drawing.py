import turtle
import random
import sys
import turtle
import random
import time
turtle.shape('classic')
turtle.color('blue', 'yellow')
turtle.speed(1000000)
f = open("colors1.txt", "r")

colors=['red']
width, height = turtle.window_width(), turtle.window_height()

minX, maxX = -width/2, width/2
minY, maxY = -height/2, height/2

screen = turtle.Screen()

turtlepower = []

turtle.tracer(0, 0)
x=0.1
z=3600
def repeated_code():
    global g
    global h
    global x
    g=x
    color = random.choice(colors)
    turtle.right(g)
    turtle.pencolor('red')
    turtle.forward(80)
    color = random.choice(colors)
    turtle.pencolor('orange')
    turtle.forward(150)
    color = random.choice(colors)
    turtle.pencolor('yellow')
    turtle.forward(10000)
    turtle.color('red')
    

     # keep moving forward until we're out of window

    if not minX <= turtle.xcor() <= maxX or not minY <= turtle.ycor() <= maxY:
        turtle.penup()
        
        turtle.setpos(0,0)  # if we don't exit() a Terminator will be raised
        turtle.pendown()
def repmore():
    global z
    for i in range(z):
        repeated_code()

    for i in range(z):
        turtle.stamp()

    turtle.update()
for i in range(1):
    repmore()
    x=0.1
    
turtle.bgcolor('yellow')
