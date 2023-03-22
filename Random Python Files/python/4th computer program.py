import time
import random
password2='2007'
gg=True
while gg == True:
    time2=5
    strike=0

    time3=time2
    password=input('enter password')
    if password==password2:
        print('correct')
        gg=False

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




    answer=input("     a - vote     b - clock c - calculator     d - change password      e - random number generator     f - exit")
   

    if answer == 'a':
        print(" vote")


    elif answer == 'b':
        print("clock")
        x=True
        while x==True:
            from datetime import datetime

            now = datetime.now()

            mm = str(now.month)

            dd = str(now.day)

            yyyy = str(now.year)

            hour = str(now.hour)

            mi = str(now.minute)

            ss = str(now.second)

            print (mm + "/" + dd + "/" + yyyy + " " + hour + ":" + mi + ":" + ss)

                
    elif answer == 'c':
        print("calculator")
        print("1. Addition");
        print("2. Subtraction");
        print("3. Multiplication");
        print("4. Division");
        print("5. Exit");
        choice = int(input("Enter your choice: "));
    if (answer>=1 and answer<=4):
        print("Enter two numbers: ");
        num1 = int(input());
        num2 = int(input());
        if choice == 1:
            res = num1 + num2;
            print("Result = ", res);
        elif choice == 2:
            res = num1 - num2;
            print("Result = ", res);
        elif choice == 3:
            res = num1 * num2;
            print("Result = ", res);
        else:
            res = num1 / num2;
            print("Result = ", res);
    elif choice == 5:
        exit();
    else:
        print("Wrong input..!!");


    elif answer == 'd':
        print("change password")
        password3=input ('do you want to change password')
    if password3=='yes':
        z=input ('enter new password')
        p=input ('enter old password')
        if p=='2007':
            password2=z
            print ('password changed')
        else:
            password2='2007'
            print ('password not changed')
    
        


    elif answer == 'e':
        print("random number generator")
        import random
        min = 1
        max = 6

        roll_again = "yes"

        while roll_again == "yes" or roll_again == "y":
            print ("Rolling the dices...")
            print ("The values are....")
            print (random.randint(min, max))
            print (random.randint(min, max))

            roll_again = raw_input("Roll the dices again?")


