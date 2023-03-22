




import sys
from turtle import *
import random
import turtle
width, height = turtle.window_width(), turtle.window_height()

minX, maxX = -width/2, width/2
minY, maxY = -height/2, height/2

turtle.speed(1)

shape('classic')
color('blue', 'yellow')

begin_fill()


while True:
    q=random.randint(1,500)
    z=random.randint(1,500)
    x=random.randint(1,500)
    v=random.randint(1,500)
    
    forward(1000)
    left(x)
    back(z)
    
    right(v)
    if not minX <= turtle.xcor() <= maxX or not minY <= turtle.ycor() <= maxY:
       
        turtle.setpos(0,0)  
     
    
    


        
end_fill()
done()
