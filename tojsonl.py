import sys
import json
import pandas as pd

if __name__ == '__main__':
    infile = "./data/dev_v2.1.json"
    outfile = "./data/dev_small_1000.json"
    df = pd.read_json(infile)
    count = 0
    with open(outfile, 'w') as f:
        for row in df.iterrows():
            data = row[1].to_json()
            parsed = json.loads(data)
            if str(parsed['answers']) == "No Answer Present.":
                continue
            print(parsed['answers'])
            count += 1
            if count == 1001:
                break
            f.write(row[1].to_json() + '\n')