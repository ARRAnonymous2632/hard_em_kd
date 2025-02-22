import random

class Dataset:
    def _read_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = f.readlines()
        data = [line.strip() for line in data]
        return data
    
    def __init__(self, S, H, T, instances_per_prompt=20, direct_translation=False, use_ground_truth=True, shuffle=True):
        S = self._read_file(S)
        H = self._read_file(H)
        T = self._read_file(T)

        assert len(S) == len(H)
        
        self.data = []
        self.instances_per_prompt = instances_per_prompt
        for i in range(len(S)):
            if H[i] == T[i] and not direct_translation:
                continue

            self.data.append([S[i]])
            if not direct_translation:
                self.data[-1].append(H[i])
            if use_ground_truth:
                self.data[-1].append(T[i])
            self.data[-1].append(i)
        
        if shuffle:
            random.shuffle(self.data)
        
        for idx in range(len(self)):
            self.data[idx] = [str(idx)] + self.data[idx]
        
        pass
        
        
        
    def __getitem__(self, idx):
        return "|||".join(self.data[idx][:-1]), self.data[idx][-1]

    def __len__(self): 
        return len(self.data)
    
    def get_batch_count(self):
        return (len(self) // self.instances_per_prompt) + 1

    def get_batch(self, batch_idx):
        start_idx = batch_idx * self.instances_per_prompt
        end_idx = min(start_idx + self.instances_per_prompt, len(self))

        return [self[i][0] for i in range(start_idx, end_idx)], [self[i][1] for i in range(start_idx, end_idx)]




