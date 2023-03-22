import random
import time

n=random.randrange(1,2)
while n<2:
    n=random.randrange(1,2)
    teachers=["mr white","mrs ahmad",'mr cox']
    rteach=random.choice(teachers)
    insult=['potato','muppet','rotton swine']
    rinslut=random.choice(insult)
    
    print(rteach,' is a ',rinslut)
    time.sleep(2)
