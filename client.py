import flwr as fl
import torch
import numpy as np
import argparse
from collections import OrderedDict
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Import modules from the conti748 project
from modules.LinkPredictor import LinkPredictor
from modules.data_pipeline.RulesDataset import RulesDataset
from modules.data_pipeline.GraphDataset import GraphDataset
from modules.utils.utils import get_config
from modules.utils.sampling import positive_sampling, construct_negative_graph

# Import the custom loss function from the original train.py
from train import compute_loss

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensor_new.cpp.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*RulesDataset.py.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Define the Flower Client ---

class GraphClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, client_id, local_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader # We don't use this, but it's good to have
        self.client_id = client_id
        self.local_epochs = local_epochs

        # Get optimizer config from the original train.py
        config = get_config("./cfg/training.yaml")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate_schedule']["start_value"]
        )

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy ndarrays."""
        print(f"[Client {self.client_id}]: get_parameters called")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy ndarrays."""
        print(f"[Client {self.client_id}]: set_parameters called")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        print(f"[Client {self.client_id}]: fit called")
        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0
        total_batches = 0

        # Run for the specified number of local epochs
        for epoch in range(self.local_epochs):
            desc = f"Client {self.client_id} Training (Epoch {epoch+1}/{self.local_epochs})"
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=desc)

            for batch_idx, inputs in progress_bar:
                self.optimizer.zero_grad()

                # Get rule_encoding from the train_loader's dataset
                rule_encoding = self.train_loader.dataset.rule_encoding

                # Positive and negative sampling (same as in train.py)
                g_input, g_gt = positive_sampling(inputs, inputs.batch, rule_encoding)
                negative_graph = construct_negative_graph(inputs, inputs.batch, rule_encoding)

                # Forward pass
                out = self.model(inputs.x.type(torch.FloatTensor), g_input.edge_index, g_gt.edge_index, g_input.edge_attr)
                out_negative = self.model(inputs.x.type(torch.FloatTensor), g_input.edge_index, negative_graph.edge_index, g_input.edge_attr)

                # Compute loss
                loss = compute_loss(out, out_negative, g_gt.edge_attr, negative_graph.edge_attr)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1
                progress_bar.set_postfix({"Loss": loss.item()})

            progress_bar.close()

        # Calculate average loss *per batch*
        avg_loss = total_loss / (total_batches)
        print(f"[Client {self.client_id}]: fit finished, Avg. Loss: {avg_loss}")

        # We send back the parameters, number of examples, and the average loss
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"avg_loss": avg_loss}

    def evaluate(self, parameters, config):
        """Evaluate the model (not used in this strategy)."""
        print(f"[Client {self.client_id}]: evaluate called (not in use)")
        return 0.0, 0, {"accuracy": 0.0}

# --- Main Client Logic ---

def main():
    parser = argparse.ArgumentParser(description="Flower Client for Graph Recommendation")
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="Client ID (from 0 to num_clients-1)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        required=True,
        help="Total number of clients in the simulation",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs to train",
    )
    args = parser.parse_args()

    # --- 1. Load and Partition Data ---
    print(f"[Client {args.client_id} of {args.num_clients}]: Loading and partitioning data...")
    config = get_config("./cfg/training.yaml")

    # Load the full training dataset
    rule_dataset = RulesDataset(config)
    rule_train, _, _ = rule_dataset.split_dataset(
        rule_dataset.dataset,
        config['training']['split']['train'],
        config['training']['split']['val']
    )

    # Partition the training data
    all_user_ids = list(rule_train.keys())
    # Use numpy.array_split to divide the user IDs into N chunks
    client_user_id_chunks = np.array_split(all_user_ids, args.num_clients)

    # Get the user IDs for *this* client
    client_user_ids = client_user_id_chunks[args.client_id]

    # Create a new dataset dictionary containing only this client's users
    client_rule_train = {user_id: rule_train[user_id] for user_id in client_user_ids}

    # Create the GraphDataset and DataLoader for this client's data
    graph_dict_train = rule_dataset.create_graph_dataset(client_rule_train)
    # --- GEMINI FIX: Pass the full rule_dataset object here ---
    graph_generator_train = GraphDataset(client_rule_train, graph_dict_train, rule_dataset)

    # We can use a larger batch size since we have fewer users per client
    # Or keep it the same as the full dataloader for consistency
    train_loader = DataLoader(
        graph_generator_train,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    print(f"[Client {args.client_id}]: Data loaded. Training on {len(client_user_ids)} users.")

    # --- 2. Instantiate Model and Client ---
    model = LinkPredictor()
    client = GraphClient(model, train_loader, None, args.client_id, args.local_epochs)

    # --- 3. Start Client ---
    print(f"Starting Flower client {args.client_id}...")
    fl.client.start_client(
        server_address="0.0.0.0:8080",
        client=client,
    )

if __name__ == "__main__":
    main()