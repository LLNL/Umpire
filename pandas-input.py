import json
import re
import pandas as pd

# Open Replay file and read it
with open('umpire.0000.replay') as f:
    lines = f.readlines()

# Do something with input
for l in lines:
    df = pd.read_json(l)
    print(df.to_string())
    
# Check input is not empty
if not df:
    print("Warning! pandas df is empty! Something went horribly wrong...")

# End program and close file
f.close()
