def rotate(l, n):
    return l[n:] + l[:n]
def bruteforce(alfa):
    
    codedword=input('enter coded word')
    codedword=codedword.lower()
    r=1
    for i in range(26):
        shift=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
        decoded=[]
        i=0
        shift.remove(' ')
        shift=rotate(shift,r)
        shift.append(' ')
        splitcodedword=[char for char in codedword]
        for char in splitcodedword:
            letter=splitcodedword[i]
            i=i+1
            searched=shift.index(letter)
            letter=alfa[searched]
            decoded.append(letter)
        decodedjoin=''.join(decoded)
        
        print(decodedjoin)
        r=r+1
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
    uncoded=uncoded.lower()
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
    ranalfa.remove(' ')
    random.shuffle(ranalfa)
    ranalfa.append(' ')
    print(ranalfa)
    return ranalfa
def decoder(alfa,ranalfa):
    i=0
    decoded=[]
    codedword=input('enter coded word')
    codedword=codedword.lower()
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
def encode_shift_cyfer(alfa):
    global i
    global mixedcoded
    global uncoded
    global splitword
    global letters
    global searched
    i=0
    mixedcoded=[]
    shifter=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    uncoded=input('input text you want coded')
    shift=int(input('number of shift'))
    shifter.remove(' ')
    shifter=rotate(shifter,shift)
    shifter.append(' ')
    uncoded=uncoded.lower()
    splitword=[char for char in uncoded]
    print(splitword)
    for char in splitword:
        letter=splitword[i]
        i=i+1
        searched=alfa.index(letter)
        print (searched)
        letter=shifter[searched]
        
        
        mixedcoded.append(letter)
        
    print(mixedcoded)
    coded=''.join(mixedcoded)
    print(coded)

alfa=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
ranalfa=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
print(alfa)
import random
ranalfa.remove(' ')
random.shuffle(ranalfa)
ranalfa.append(' ')
print(ranalfa)
while True:
    option=input('''
    1. randomise cyfer
    2. code a word
    3. decode a word
    4. brute force a shift cyfer
    5. encode with shift cyfer
    6. end
    ''')
    if option=='1':
        ranalfa=randomizecyfer(alfa,ranalfa)
    elif option=='2':
        coder(alfa,ranalfa)
    elif option=='3':
        decoder(alfa,ranalfa)
    elif option=='4':
        bruteforce(alfa)
    elif option=='5':
        encode_shift_cyfer(alfa)
    elif option=='6':
        exit()

