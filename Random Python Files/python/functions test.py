


def rectangle (length, width):
    area = length * width
    perimeter =(length*2)+(width*2)
    print ('press 1 for rectangle diameter or')
    print ('press 2 for rectangle perimiter')
    choice= input ()
    if choice =='1':
        print('the area of the rectangle is ',area)
    elif choice =='2':
        print('the perimeter of the rectangle is ',perimeter)
    else:
        print ('invalid choise')


def password():
    password=2007
    attempts = 3
    while attempts >0:
        guess = int(input("ENTER PASSWORD"))
    
        attempts=attempts-1
        
        if guess ==password:
            print ("RIGHT")
            attempts=0
        else:
            print ('WRONG')
            if attempts==0:
                n=1
                n=n*2
                print('you have been locked out for '+str(n))
            else:
                print (' ')
            
            

    


password()
length = int (input ('enter length'))
width =int (input ('enter width'))
rectangle(length, width)
