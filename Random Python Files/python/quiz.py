
x=True
while x==True:
    m=0
    name=input('what is your name')
    print ('hello '+name+' i am computer quiz')
    quiz=input('do you want to do a quiz?')
    
    if quiz=='yes' or'Yes':
        print('oh goodie')
    elif quiz=='no'or 'No' or'nO' or'NO':
        print('but I want to '+name+' let\'s do it anyway')
    else:
        print('oh goodie')
        m=m+1
    print ('question 1')
    n=name+'computerquiz'

    q1=input('what is your name plus my name');
    if q1==n:
        print('correct funny name '+n+' isn\'t it')
        m=m+1

    else:
        print('wrong')


    q2=input('what do you get when you cross a shark with ice');
    if q2=='frostbite'or'frost bite'or'Frostbite' or 'Frost Bite' or 'frost Bite':
        print('correct ha ha ha ha')
        m=m+1
    else:
        print('wrong')


    q3=input('what is my age');
    if q3=='I do not know' or 'i dont know' or '3 weeks':
        print('correct or 3 weeks')
        m=m+1
       
    else:
        print('wrong')

    abc=name+'3weeks'
    q4=input('what is your name plus my age');
    if q4==abc:
        print('correct well done')
        m=m+1
        
    else:
        print('wrong')


    ac='1'+'bob'
    q5=input('what is 1+bob');
    if q5==ac:
        print('correct well done')
        m=m+1

    else:
        print('wrong')

    print(''' the answers were:'''
        +n+'''
        frostbite
        3 weeks'''
        +abc+''' '''
        +ac)
            
            
            
    
    print(m)
    

    
