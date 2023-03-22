import time
import random
jj_password='2007'
gg=True
while gg == True:
    time2=5
    strike=0
    x=True
    while x == True:
        time3=time2
        password=input('enter password')
        if password==jj_password:
            print('correct')
            x=False
        else:
            print('incorrect')
            strike=strike+1
            if strike==3:
                strike=0
                print ('you have been locked out for',time2)
                for y in range(time2):
                    time.sleep(1)
                    time3=time3-1
                    print(time3)
                time2=time2*2
    pwd=input ('do you want to change password')
    if pwd=='yes':
            z=input ('enter new password')
            p=input ('enter old password')
            if p=='2007':
                jj_password=z
                print ('password changed')
            else:
                jj_password='2007'
                print ('password not changed')
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    g=0
    h=0
    repeat=True
    while repeat==True:
        vote= input ("vote A=DONALD TRUMP,B=THERRESA MAY,C=BORIS JOHNSON,d=DAVID CAMERON,e=EDWARD SMITH,f=FINLY FORD g=danny or h=JACK MARCHANT").lower()
        if vote=="a":
            a=a+1
        elif vote=="b":
            b=b+1
        elif vote=="c":
            c=c+1
        elif vote=="d":
            d=d+1
        elif vote=="e":
            e=e+1
        elif vote=="f":
            f=f+1
        elif vote=="g":
            g=g+1
        elif vote=="h":
            h=h+1
        end=input ("do you want to see the results so far").lower()
        if end== "yes":
            print ("DONALD TRUMP=",a)
            print ("THERRESA MAY=",b)
            print ("BORIS JOHNSON=",c)
            print ("DAVID CAMERON=",d)
            print ("EDWARD SMITH=",e)
            print ("finley ford=",f)
            print ("danny=",g)
            print ("JACK MARCHANT=",h)
        if end=='no':
            repeat=False
        else:
            vote
