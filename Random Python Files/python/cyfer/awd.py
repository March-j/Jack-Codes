import time

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)
x=0
mem=x
#loop=403291461126605635584000000
#loop=4.0329146112661E+24
loop=25715345
half=loop/2
for i in range(int(loop)):
    x=x+1
    if x == loop:
        break
    # if x>0:
    #     mem=x
    #     x=x-x
    # else:
    #     x=mem
    #     x=x+1
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)
x=0
print(x)
