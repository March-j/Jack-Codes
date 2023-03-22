from turtle import *
color('red', 'yellow')
begin_fill()
x=10
while True:
    x=x+1
    speed(999)
    forward(50)
    left(x)
    if abs(pos()) < 0:
        break
end_fill()
done()
