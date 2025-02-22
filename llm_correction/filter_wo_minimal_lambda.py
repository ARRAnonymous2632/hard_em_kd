
import Levenshtein
from utils import read_file, write_lines, get_loss_dict
from tqdm import tqdm

def main(args):
    loss_data_pairs = [
        (   "loss_scores/base_itr1_42_loss_values.log", \
            "data/batch_inference/base_itr1_42/data/kd.out.shard.out.H.modified.2" 
        ),
        (   "loss_scores/base_itr1_37_loss_values.log", \
            "data/batch_inference/base_itr1__37/data/kd.out.shard.out.H.modified.2" 
        ),
        (   "loss_scores/base_itr1_97_loss_values.log", \
            "data/batch_inference/base_itr1_97/data/kd.out.shard.out.H.modified.2" 
        )
    ]
    H = read_file("data/base/kd.out.shard.out.H.2")
    T = read_file("data/base/kd.out.shard.out.T.2")

    loss_data_pairs = [
        (get_loss_dict(l_path), read_file(d_path)) for l_path, d_path in loss_data_pairs
    ]
    
    rs = [[Levenshtein.ratio(data[i].split(), H[i].split()) for i in range(len(H))] for _, data in loss_data_pairs]
    losses_avg = [sum(d.values()) / len(d) for d, _ in loss_data_pairs]

    rs_avg = sum([sum(r) / len(H) for r in rs]) / len(loss_data_pairs)
    losses_avg = sum(losses_avg) / len(loss_data_pairs)

    lambda_ = losses_avg / rs_avg
    
    print(f"Lambda: {lambda_}")
    print(f"R: {[sum(r) / len(H) for r in rs]}")
    print(f"L: {losses_avg}")

    for factor in [-3 * lambda_, 3 * lambda_]:
        chosen_ones = []
        failures = 0
        for i in tqdm(range(len(H))):
            best_sent = None
            best_score = float("-inf")
            for loss_dict, data in loss_data_pairs:
                if i not in loss_dict:
                    continue 
                lev_ratio = Levenshtein.ratio(data[i].split(), H[i].split())
                # score = -loss + factor * ratio 
                if factor == "inf":
                    score = lev_ratio
                elif factor == "-inf":
                    score = -lev_ratio
                else:
                    score = lev_ratio * factor - loss_dict[i]

                if score > best_score:
                    best_sent = data[i]
                    best_score = score 
            if best_sent:
                chosen_ones.append(best_sent)
            else:
                chosen_ones.append(T[i])
                failures += 1
        
        write_lines(args.out + f".lambda.{factor}", chosen_ones)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)

    args = parser.parse_args()
    
    main(args)
            
            