import random
def n1():
    x=random.randint(1,12)
    
def n2():
    q=random.randint(1,6)
  
def n3():
    p=random.randint(1,4)
 
def n4():
    p=random.randint(1,4)
    x=random.randint(1,12)
    n5=(x/p)
    n6=(n5+10)
    print(n6)
x=3
for x in range(3):
    chskill=[]
    chname=input('what is your charicters name')
    chskill.append('name:'+chname)
    p=random.randint(1,4)
    x=random.randint(1,12)
    n5=(x/p)
    n6=(n5+10)
    chskill.append('strength'+str(n6))
    p=random.randint(1,4)
    x=random.randint(1,12)
    n5=(x/p)
    n6=(n5+10)
    chskill.append('skill'+str(n6))

    print(chskill)
    x-1

