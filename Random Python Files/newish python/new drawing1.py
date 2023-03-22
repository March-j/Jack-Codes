# import package
import turtle 
turtle.speed(10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)
# method to raw pattern
# of circle with rad radius
def draw(rad):
      
    # draw circle
    turtle.circle(rad)
      
    # set the position by using setpos()
    turtle.up()
    turtle.setpos(0,-rad)
    turtle.down()
  
# loop for pattern
for i in range(100000000000000000000000000000000000):
    draw(1+1*i)
