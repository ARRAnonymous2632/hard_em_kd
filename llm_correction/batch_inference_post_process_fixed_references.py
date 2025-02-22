import re
import argparse
import json
import os
from utils import read_file, write_lines


def main(args):
    output_file_name = "kd.out.shard.out.T.modified.2.empty.lines"
    T = read_file(args.targets)
    S = read_file(args.sources)
    regex = r"\|+"
    data = dict()
    invalid_ids = []
    for root, dirs, files in os.walk(args.chunks_folder):
        for file in files:
            if ".json" not in file:
                continue

            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                sub_data = json.load(f)
        
            for item in sub_data:
                responses = item["response"].strip().split("\n")
                ids = item["ids"]
                lines = []
                for idx, line in enumerate(responses):
                    new_line = [part.strip() for part in re.split(regex, line) if part.strip()]
                    if len(new_line) == 3:
                        try:
                            new_line[0] = int(new_line[0]) % len(ids)
                            new_line = tuple(new_line)
                            lines.append(new_line)
                        except:
                            pass
                    
                for idx, s, r in lines:
                    data[ids[idx]] = {"S": s, "R": r}
                
                for id in ids:
                    if id not in data:
                        invalid_ids.append(str(id))
    
    print("Done loading, unifying fixed hypothesis ...")
    fixed_hypos = []
    for i in range(len(T)):
        if T[i] == S[i]:
            fixed_hypos.append(T[i])
        elif i in data:
            fixed_hypos.append(data[i]['R'])
        else:
            fixed_hypos.append("")
    
    path = os.path.join(args.output_folder, output_file_name)
    write_lines(path, fixed_hypos)
    path = os.path.join(args.output_folder, output_file_name + ".faults")
    write_lines(path, invalid_ids)

    print(f"Finished, invalid IDs: {len(invalid_ids)}")
            
                
                

            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunks_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--targets", required=True)
    # parser.add_argument("--hypos", required=True)
    parser.add_argument("--sources", required=True)


    args = parser.parse_args()
    
    main(args)