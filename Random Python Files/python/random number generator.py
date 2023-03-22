import random
def n1():
    n=random.randint(1,12)
    print(n)
def n2():
    n=random.randint(1,6)
    print(n)
def n3():
    n=random.randint(1,4)
    print(n)
def n4():

    x=True
    while x==True:
        sides=input('how many sides do you want your dice to have 4,6,12 (to end=x)')
        elif sides=='x':
             exit()
        rep=int(input('how many times do you want ot repeat?'))
        while rep >0:
            rep=rep-1
            if sides=='4':
                n3()
            elif sides=='6':
                n2()
            elif sides=='12':
                n1()


