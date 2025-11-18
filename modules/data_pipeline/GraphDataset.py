import torch

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, graph_dict, rule_dataset_obj=None):
        self.data = data
        self.graph_dict = graph_dict

        # --- GEMINI FIX ---
        # We need the full rule_dataset object to access the rule_encoding
        # This will be passed in by our client.
        if rule_dataset_obj:
            self.rule_encoding = rule_dataset_obj.rule_encoding
        # --- END FIX ---

        self.userid2idx = {j: k for j, k in enumerate(data.keys())}
        self.idx2userid = {k: j for j, k in enumerate(data.keys())}

        # This re-indexing is fine, but we must use the correct keys
        # We'll use the indices (0, 1, 2...) as the keys for __getitem__
        self.data = {k: v for k, v in data.items()}
        self.graph_dict = {k: v for k, v in graph_dict.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use the integer-based index to get the user_id
        # --- GEMINI FIX for KeyError ---
        # The original code had userid2idx and idx2userid swapped.
        # self.userid2idx is the one that maps an int (idx) to a user_id (str)
        user_id = self.userid2idx[idx]
        return self.graph_dict[user_id]