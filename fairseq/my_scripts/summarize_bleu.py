import os
import sys
import json
path = sys.argv[1]
out_path = sys.argv[2]

file_list = os.listdir(path)

def get_bleu(file_name, is_json=False):
    if is_json == False:
        file_lines = open(os.path.join(path, file_name)).readlines()
        for line in file_lines:
            if line.startswith("BLEU"):
                segment_list = line.split(' ')
                return float(segment_list[2])
    else:
        data = json.load(open(os.path.join(path, file_name)))
        return float(data['score'])
        


bleu_dict = {}
for file_name in file_list:
    if file_name.startswith('bleu'):
        file_name_body, _ = file_name.split('.')
        _, src, trg = file_name_body.split('_')
        bleu_dict[f'{src}-{trg}'] = get_bleu(file_name, is_json=True)

# print('lllll')
# print(bleu_dict)

bleu_dict_sorted = dict(sorted(bleu_dict.items(), key=lambda x: x))

out_file = open(out_path, 'w')
out_file.write("AVG" + ' ')

for key in bleu_dict_sorted:
    out_file.write(key + ' ')

out_file.write('\n')

mean = sum(bleu_dict_sorted.values()) / len(bleu_dict_sorted.values())
out_file.write(str(round(mean, 2)) + ' ')

for key in bleu_dict_sorted:
    out_file.write(str(round(bleu_dict_sorted[key], 2)) + ' ')

out_file.write('\n')

