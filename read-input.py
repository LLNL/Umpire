import re
import pandas as pd

# Open Replay file and read it
global kind, uid, timestamp, event, payload
names = ["kind", "uid", "timestamp", "event", "payload"]
replay_lines = dict() 

for n in names:
    replay_lines[n] = list()

print(replay_lines)

with open('umpire.0000.replay') as f:
    for line in f:
        replay_lines["kind"].append(re.findall(".kind.:.\w+.", line))
        replay_lines["uid"].append(re.findall(".uid.:.\w+.", line))
        replay_lines["timestamp"].append(re.findall(".timestamp.:.\w+.", line))
        replay_lines["event"].append(re.findall(".event.: \"\w+\"", line))
        replay_lines["payload"].append(re.findall(".payload.:( \{ \".*\" \}| \{ .*\d+ \})", line))
        
# Make sure kind/uid/timestamp/event/payload are not empty, else print warning
for key, value in replay_lines.items():
    if not value:
        print("Warning!" + str(key) + "is empty. Error probably occurred.") 

# Do other stuff with this info?
df = pd.DataFrame(replay_lines)
print(df)

# End program and close file
f.close()
