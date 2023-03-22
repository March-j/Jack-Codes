import glob, os
import shutil
import time
import fnmatch
score=0
filen=open("filenames.txt", "a+")
def finding():
    global leng
    global file
    
    global filen
    path = r"C:\Users\The Marchants\Documents\0 Jacks\python text test"
    file = fnmatch.filter(os.listdir(path), "*.txt")
    print(score)
    filen.write(str(file))
    print(filen.read())
    
    

def moving():
    global filen
    x=filen.read()
    original = os.path.join(r'C:\Users\The Marchants\Documents\0 Jacks\python text test',x)
    target = r'C:\Users\The Marchants\Documents\0 Jacks\New folder'
    print(filen)
    shutil.move(original,target)
    
def check():
    global filen
    finding()
    if filen.read == '':
           exit()


check()     #checks if there is any of the right files in there 
while True:
    finding()
    moving()
    print(file)

           #finds the files
    
    

    
           # moves the files

exit()

    
