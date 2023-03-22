from turtle import *
color('blue', 'yellow')
begin_fill()

x=200
z=99
while True:
    x=x+0.1
    z=z+1
    speed(99)
    forward(x)
    right(z)

    if abs(pos()) <0.000000001:
        break
end_fill()
done()
