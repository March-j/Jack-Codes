x=1
for x in range(2):
    z=input('input your word')
    b=z.count('a')
    c=z.count('A')
    a=b+c
    print('your word has ',a,' a\'s')
    y=input('use again?')
    if y=='yes'or'Yes':
        print('ok')
    else:
        x=5
