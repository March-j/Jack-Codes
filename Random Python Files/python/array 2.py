    
# first line is setting up an array
dolist= ["walk the car","wash the dog","Eat slime"]
print (dolist)
x='yes'


# takes input from user to see if they want to add to list
while x=='yes':
    additem =input ("would you like to add an item to the list")
    # if user input is yes
    if additem =='yes':
    #adds an item
      addit = input("which item would you like to add?")
      dolist.append (addit)
    #removes item
    else:
      removeitem= input ("would you like to remove an item from the list")
      if removeitem =="yes":
        removeit = input("which item would you like to remove?")
        dolist.remove (removeit)
    x=input ('do you want to do anything else')
    
        
    #shows the list
print (dolist)
leng=len(dolist)
print(leng)

