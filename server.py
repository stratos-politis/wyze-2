import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import itertools
from torch_geometric.data import Data

# Import modules from the conti748 project
from modules.LinkPredictor import LinkPredictor
from modules.data_pipeline.RulesDataset import RulesDataset
from modules.data_pipeline.GraphDataset import GraphDataset
from modules.utils.utils import get_config, find_rank
from modules.utils.decode_results import decode_output
from torch_geometric.loader import DataLoader

# Import the custom loss function from the original train.py
from train import compute_loss

# Import the sampling functions
from modules.utils.sampling import positive_sampling, construct_negative_graph

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensor_new.cpp.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*RulesDataset.py.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Configuration ---
NUM_ROUNDS = 50
NUM_CLIENTS = 10 # Number of clients to wait for

# --- Server-Side Evaluation (Centralized Evaluation) ---
print("Loading centralized evaluation data for the server...")
config = get_config("./cfg/training.yaml")
# We load the full dataset info to build the val/test sets
rule_dataset = RulesDataset(config)
rule_train, rule_val, rule_test = rule_dataset.split_dataset(
    rule_dataset.dataset,
    config['training']['split']['train'],
    config['training']['split']['val']
)

# Create Validation DataLoader
graph_dict_val = rule_dataset.create_graph_dataset(rule_val)
graph_generator_val = GraphDataset(rule_val, graph_dict_val, rule_dataset) # Pass rule_dataset
val_loader = DataLoader(graph_generator_val, batch_size=512, shuffle=False)

# Create Test DataLoader (for MRR)
graph_dict_test = rule_dataset.create_graph_dataset(rule_test)
graph_generator_test = GraphDataset(rule_test, graph_dict_test, rule_dataset) # Pass rule_dataset
print("Server evaluation data loaded.")

# Define the global model (used for evaluation)
model = LinkPredictor()

def get_model_parameters(model):
    """Get model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """Set model parameters from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# --- Define the server-side evaluation function (calculates Loss and MRR) ---
def get_evaluate_fn(val_loader, test_graph_generator, rule_dataset_obj):
    """
    This function will be called by the strategy to evaluate the
    global model (after aggregation) on the server's validation and test sets.
    """
    def evaluate(server_round, parameters, config):
        # 1. Set the global model parameters
        set_model_parameters(model, parameters)
        model.eval()

        # === Part 1: Calculate Validation Loss ===
        val_loss = 0.0
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for val_inputs in tqdm(val_loader, desc=f"Server Validating (Round {server_round})"):
                g_input, g_gt = positive_sampling(val_inputs, val_inputs.batch, rule_dataset_obj.rule_encoding)
                negative_graph = construct_negative_graph(val_inputs, val_inputs.batch, rule_dataset_obj.rule_encoding)

                out = model(val_inputs.x.type(torch.FloatTensor), g_input.edge_index, g_gt.edge_index, g_input.edge_attr)
                out_negative = model(val_inputs.x.type(torch.FloatTensor), g_input.edge_index, negative_graph.edge_index, g_input.edge_attr)

                loss = compute_loss(out, out_negative, g_gt.edge_attr, negative_graph.edge_attr)
                val_loss += loss.item()

        avg_val_loss = val_loss / num_val_batches

        # === Part 2: Calculate Test MRR ===
        # This is the logic from the centralized train.py's test loop
        ranks = []
        num_user = 0
        with torch.no_grad():
            for idx in tqdm(range(len(test_graph_generator)), desc=f"Server Testing MRR (Round {server_round})"):

                # --- GEMINI FIX for KeyError: 0 ---
                # We get the user_id string using the *correct* dictionary
                user_id = test_graph_generator.userid2idx[idx]
                data_test = test_graph_generator[idx]
                # --- End Fix ---

                num_nodes = len(data_test.x)
                if num_nodes == 0:
                    continue # Skip empty graphs

                num_user += 1

                # Use positive_sampling to get the leave-one-out ground truth
                g_input_test, g_gt_test = positive_sampling(data_test, torch.zeros(num_nodes, dtype=torch.long), rule_dataset_obj.rule_encoding)

                # Create all possible edges for prediction
                all_possible_edges = list(itertools.product(range(num_nodes), repeat=2))
                src = [i[0] for i in all_possible_edges]
                dst = [i[1] for i in all_possible_edges]
                edge_index_comb = torch.tensor([src, dst], dtype=torch.long)
                g_combination = Data(x=data_test.x, edge_index=edge_index_comb)

                # Get model output
                out = model(g_input_test.x, g_input_test.edge_index, g_combination.edge_index, g_input_test.edge_attr)

                # Decode the output to get top-50 recommendations
                node_mapping = rule_dataset_obj.node_mapping[user_id]
                results_list = decode_output(g_input_test, all_possible_edges,
                                               out, rule_dataset_obj.rule_encoding,
                                               node_mapping)

                # Get the ground truth tuple
                node_decoding = {v: k for k, v in node_mapping.items()}
                rule_decoding = {v: k for k, v in rule_dataset_obj.rule_encoding.items()}

                gt_tuple = (node_decoding[g_gt_test.edge_index[0].item()],
                            node_decoding[g_gt_test.edge_index[1].item()],
                            rule_decoding[g_gt_test.edge_attr[0].item()])

                # Find the rank of the ground truth in the recommendations
                r = [(res[0], res[1], res[2]) for res in results_list]
                rank = find_rank(r, gt_tuple)
                if rank is not None:
                    ranks.append(1.0 / rank)

        # Calculate final MRR
        mrr_score = np.sum(ranks) / num_user if num_user > 0 else 0.0

        print(f"Server-side evaluation, Round {server_round}: Avg. Val. Loss: {avg_val_loss}, MRR: {mrr_score}")

        # Return loss and metrics
        return avg_val_loss, {"val_loss": avg_val_loss, "mrr": mrr_score}

    return evaluate

# --- Define the Strategy ---
print(f"Starting Flower server for {NUM_ROUNDS} rounds, waiting for {NUM_CLIENTS} clients...")

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # Use 100% of connected clients for training
    fraction_evaluate=0.0,      # Do NOT ask clients to evaluate
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,

    # Use our server-side evaluation function
    evaluate_fn=get_evaluate_fn(val_loader, graph_generator_test, rule_dataset),

    # Pass the initial model parameters to the server
    initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters(model)),
)

# --- Start the Server ---
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)