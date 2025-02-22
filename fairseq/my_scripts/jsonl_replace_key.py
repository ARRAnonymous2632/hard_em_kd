import json
import sys
import os

def replace_key_in_jsonl(input_file):
    output_file = os.path.splitext(input_file)[0] + "_modified.jsonl"
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            if 'prediction' in data:
                data['hypothesis'] = data.pop('prediction')
            outfile.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python jsonl_replace_key.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    replace_key_in_jsonl(input_file)