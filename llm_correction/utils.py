import pickle

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    return data

def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(lines))


def get_loss_dict(path):
    """
    Reads the file created by fairseq to get a dictionary for loss values of each index. The input file sample is like:
    LOSS-266574	2.051720380783081
    LOSS-306061	1.6525481939315796
    """
    def parse_line(line):
        id_loss_pair = line.split("-")[1].split("\t")

        id = int(id_loss_pair[0])
        loss = float(id_loss_pair[1])
        return id, loss

    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    losses = {}
    for line in lines:
        if line.startswith("LOSS-"):
            id, loss = parse_line(line)
            losses[id] = loss
    return losses

def load_losses_from_file(path, data):
    
    
    losses = get_loss_dict(path)
    
    idx = 0
    for id in data:
        data[id]["loss"] = []
        for _ in data[id]["sents"]:
            data[id]["loss"].append(losses.get(idx, float("inf")))
            idx += 1
        
    return data