 
x2 = True;
name = input ("what is your name?")
print("hello " +name+ " my name is mort nice to meet you")
house = input ("where do you live " +name)
print ("i like " + house + " it is very nice there " + name)
pet = input ("do you have a pet" + name)
if pet == 'yes':
    print ("good for you ")
    Type=input ("what is it ")
    if Type =='dog':
         print ("they are the best")
    elif Type == 'cat':
         print ("bad cats make me sneeze")
    else:
         print ("oh i have never heard of that before " +name)
    pet_name=input ("what is their name " + name)
    print ("ah cute " + pet_name + "is very nice")
else:
    print (" you should get one ")
while x2==True:
    answer =input (" do you want to chat " + name)
    if answer=='yes':
        print ("brilliant")
        x2=False;
    elif answer=='no':
               print ("wrong answer");
    else:
        print("wrong answer");
siblings=input("do you have a brother or a sister")
if siblings=='yes':
    print ("nice")
    gender2=input(" brother or sister")
    if gender2=='sister':
        name2=input("what is her name")
        print (name2 + " nice name")
    elif gender2=='brother':
        name3=input("what is his name")
        print (name3 + " nice name")
    else:
        name4=input("what is their names")
        print (name4 + " nice name")
elif gender=='sister':
     name2=input("what is her name")
     print (name2 + " nice name")
elif gender=='brother':
     name3=input("what is his name")
     print (name3 + " nice name")
elif gender== 'both':
     name4=input("what is their names")
     print (name4 + " nice name")
                
else:
    print ("luck you ")
subject=input("do you like maths")
if subject=='yes':
    print("good for you")
else:
    print (" how could you")
teeth=input ("do you have any baby teeth ")
if teeth=='yes':
    print ("wow they are really small")
    

else:
    print ("good for you ")
    x=True;
    while x==True:
        me=input("do you like me")
        if me=='yes':
            print (" right answer")
            x=False;
        elif me=='no':
            print ("wrong");
        else:
            print ("wrong");
print ("shutting down")
exit()
