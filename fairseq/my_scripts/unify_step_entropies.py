import random



def main():
    random.seed(37)
    paths = [
        ("raw", "checkpoints/ckpt_llm_nat/checkpoints_wmt14ende_raw_at_6_6_student_revised/for_eval/_/wmt14.en-de_raw_loss_values.log"),
        ("kd", "checkpoints/ckpt_llm_nat/checkpoints_baseline_kd/for_eval/_/wmt14_en_de_raw_at_student_6_6_revised_loss_values.log"),
        ("fixed_fixed", "checkpoints/ckpt_llm_nat/checkpoints_wmt14ende_raw_at_6_6_student_revised_qwen_fixed_fixed/for_eval/_/wmt14_en_de_raw_at_student_6_6_revised_qwen_fixed_fixed_loss_values.log"),
        ("fixed", "checkpoints/ckpt_llm_nat/checkpoints_wmt14ende_raw_at_6_6_student_revised_qwen_fixed/for_eval/_/wmt14_en_de_raw_at_student_6_6_revised_qwen_fixed_loss_values.log")
    ]

    data = dict()
    def parse_line(line):
        values = line.split("\t")
        id = values[0].split("-")[-1]
        id = int(id)
        values = [float(v) for v in values[1:]]
        return id, values
    indices = None
    for key, path in paths:
        data[key] = dict()
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.startswith("ENT")]
        if not indices:
            indices = random.sample(range(len(lines)), 120000)
        from tqdm import tqdm
        for line in tqdm(lines):
            id, values = parse_line(line)
            data[key][id] = values
    import pickle
    new_data = dict()
    for key in data:
        new_data[key] = {idx: data[key].get(idx, None) for idx in indices}
    
    with open("entropies.pkl", "wb") as f:
        pickle.dump(new_data, f)
    

if __name__ == "__main__":
    main()