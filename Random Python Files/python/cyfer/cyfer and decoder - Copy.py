def coder(alfa,ranalfa):
    global i
    global mixedcoded
    global uncoded
    global splitword
    global letters
    global searched
    i=0
    mixedcoded=[]
    uncoded=input('input text you want coded')
    splitword=[char for char in uncoded]
    print(splitword)
    for char in splitword:
        letter=splitword[i]
        i=i+1
        searched=alfa.index(letter)
        print (searched)
        letter=ranalfa[searched]
        
        
        mixedcoded.append(letter)
        
    print(mixedcoded)
    coded=''.join(mixedcoded)
    print(coded)

def randomizecyfer(alfa,ranalfa):
    random.shuffle(ranalfa)
    print(ranalfa)
    return ranalfa
def decoder(alfa,ranalfa):
    i=0
    decoded=[]
    codedword=input('enter coded word')
    splitcodedword=[char for char in codedword]
    for char in splitcodedword:
        letter=splitcodedword[i]
        i=i+1
        searched=ranalfa.index(letter)
        print (searched)
        letter=alfa[searched]
        decoded.append(letter)
    print(decoded)
    decodedjoin=''.join(decoded)
    print(decodedjoin)
alfa=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ranalfa=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
print(alfa)
import random
f=open('cyfer.txt','r')

alfaread=f.read()
print(alfaread)
ranalfa = alfaread.split(' ')




while True:
    option=input('''
    1. randomise cyfer
    2. code a word
    3. decode a word
    ''')
    if option=='1':
    
        ranalfa=randomizecyfer(alfa,ranalfa)
        ranalfajoin=' '.join(ranalfa)
        f = open("cyfer.txt", "w")
        f.write(ranalfajoin)
        f.close()
    elif option=='2':
        coder(alfa,ranalfa)
    elif option=='3':
        decoder(alfa,ranalfa)
    

