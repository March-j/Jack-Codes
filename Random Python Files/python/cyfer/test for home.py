alfa=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','x','y','z']
#f=open('cyfer.txt','w')
#alfajoin=(' '.join(alfa))
#f.write(alfajoin)
#f.close()


f=open('cyfer.txt','r')

alfaread=f.read()
print(alfaread)
x = alfaread.split(' ')

print(x)
import random
random.shuffle(alfa)
print(alfa)
print(' '.join(alfa))
