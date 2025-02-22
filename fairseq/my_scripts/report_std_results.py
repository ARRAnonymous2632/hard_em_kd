import os
import re
import numpy as np


def read_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

import time
def main(folder_path):
    bleu = []
    sacrebleu = []
    chrf = []
    comet = []
    time.sleep(5)
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if os.path.isfile(path):
            if file.endswith(".out"):
                lines = read_file(path)
                score = lines[-1]
                match = re.search(r'BLEU4\s*=\s*([\d.]+)', score)
                if match:
                    bleu.append(float(match.group(1)))
            elif file.endswith(".SACREBLEU.RESULT"):
                lines = read_file(path)
                score = lines[0]
                match = re.search(r'BLEU[^\s]*\s*=\s*([\d.]+)', score)
                sacrebleu.append(float(match.group(1)))
                score = lines[1]
                match = re.search(r'chrF[^\s]*\s*=\s*([\d.]+)', score)
                chrf.append(float(match.group(1)))
            elif file.endswith(".COMET-SCORE.RESULTS"):
                lines = read_file(path)
                score = lines[-1]
                match = re.search(r'score:\s*([\d.]+)', score)
                comet.append(float(match.group(1)) * 100)
    
    print("BLEUs: ", bleu)
    print(f"BLEU Score: {np.mean(bleu):.4f} +- {np.std(bleu):.4f}")

    print()
    
    print("SacreBLEUs: ", sacrebleu)
    print(f"SacreBLEU Score: {np.mean(sacrebleu):.4f} +- {np.std(sacrebleu):.4f}")

    print()
    
    print("CometScores: ", comet)
    print(f"Comet Score: {np.mean(comet):.4f} +- {np.std(comet):.4f}")

    
    print()
    
    print("chrF2: ", chrf)
    print(f"chrF2 Score: {np.mean(chrf):.4f} +- {np.std(chrf):.4f}")

import sys
if __name__ == "__main__":
    folder = sys.argv[1]

    main(folder)