from turtle import *
shape('classic')
color('blue', 'yellow')
begin_fill()



import random
import time
def more():
    global x
    global z
    x=x+1
    z=z+1
    forward(x)
    left(z)
    

def less():
    global x
    global z

    x=x-1
    z=z-1
    forward(x)
    left(z)
    
x=2
z=2
while True:
    
    q=random.randint(0,1)
    
    if q==1:
        less()
        print('less')
    else:
        more()
        print('more')
    if abs(pos()) < 0.0000001:
        break


    
end_fill()
done()
