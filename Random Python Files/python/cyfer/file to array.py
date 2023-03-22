import pandas as pd
repeatsofcode=0
def file_to_list(file):
    global repeatsofcode
    repeatsofcode=repeatsofcode+1
    print(repeatsofcode)
    rtn: object = []
    file_object: object = open(file, "r")
    rtn: object = file_object.read().splitlines()
    file_object.close()
    return list(filter(None, pd.unique(rtn).tolist())) # Remove Empty/Duplicates Values
    pass

# Example #
while True:
    file=input('input file name')
    data_from_file: object = file_to_list(file)    
    print(data_from_file)
    f=open('array.txt')
    f.write(data_from_file)
    f.close()
