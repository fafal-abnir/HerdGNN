import itertools

base_command = "python3 dyfraudnet_main.py"
fixed_args = {
    "--epochs": 200,
    "--gnn_type": "GIN",
    "--dataset_name": "RedditTitle",
    "--graph_window_size": "week",
    "--memory_size": 256,
    "--num_windows": 150,
}

learning_rates = [0.005, 0.01]
alphas = [0.0, 0.05, 0.1]
dropouts = [0.0, 0.3]
blend_factors = [0.9, 1.0]
hidden_sizes = [128]
flags = [
    [],  # no flags
    ["--enable_memory"],
    ["--fresh_start"],
    ["--enable_memory", "--fresh_start"],
]

combinations = list(itertools.product(
    learning_rates, alphas, dropouts, blend_factors, hidden_sizes, flags
))

seen = set()

for lr, alpha, dropout, blend, hidden, flag_list in combinations:
    if alpha == 0.0 and blend != 0.9:
        continue

    args = fixed_args.copy()
    args["--learning_rate"] = lr
    args["--alpha"] = alpha
    args["--dropout"] = dropout
    args["--blend_factor"] = blend
    args["--hidden_size"] = hidden

    # Optional: ensure no exact duplicates
    config_signature = tuple(sorted(args.items()) + [tuple(sorted(flag_list))])
    if config_signature in seen:
        continue
    seen.add(config_signature)

    cmd = [base_command] + [f"{k}={v}" for k, v in args.items()] + flag_list
    print(" ".join(cmd) + ';')