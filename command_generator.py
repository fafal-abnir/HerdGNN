import itertools

base_command = "python3 dyfraudnet_main.py"
fixed_args = {
    "--epochs": 200,
    "--gnn_type": "GIN",
    "--dataset_name": "MOOC",
    "--graph_window_size": "day",
    "--num_windows": 30,
}

learning_rates = [0.005]
alphas = [0.0, 0.05, 0.1]
dropouts = [0.0, 0.3]
blend_factors = [0.9, 1.0]
hidden_sizes = [32,64,128]
num_layers = [1,2]
flags = [
    [],  # no flags
    ["--enable_memory"],
    ["--fresh_start"],
    ["--enable_memory", "--fresh_start"],
]

combinations = list(itertools.product(
    learning_rates, alphas, num_layers, dropouts, blend_factors, hidden_sizes, flags
))

seen = set()

for lr, alpha, num_layer, dropout, blend, hidden, flag_list in combinations:
    if alpha == 0.0 and blend != 0.9:
        continue

    args = fixed_args.copy()
    args["--learning_rate"] = lr
    args["--alpha"] = alpha
    args["--dropout"] = dropout
    args["--blend_factor"] = blend
    args["--hidden_size"] = hidden
    args["--memory_size"] = hidden
    args["--num_layers"] = num_layer
    # Optional: ensure no exact duplicates
    config_signature = tuple(sorted(args.items()) + [tuple(sorted(flag_list))])
    if config_signature in seen:
        continue
    seen.add(config_signature)

    cmd = [base_command] + [f"{k}={v}" for k, v in args.items()] + flag_list
    print(" ".join(cmd) + ';')