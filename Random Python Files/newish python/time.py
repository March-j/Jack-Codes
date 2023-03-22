from datetime import datetime
import time
import os
x=False
while x==False:
    
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print(current_time)
    time.sleep(1)
    
    
    os.system('cls')

    

        
        
