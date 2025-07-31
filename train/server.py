# server.py
import flwr as fl

# ---- 1. Metric Aggregation Function ----
def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    def avg(metric_name):
        return sum(num_examples * m[metric_name] for num_examples, m in metrics) / total_examples

    aggregated = {
        "accuracy": avg("accuracy"),
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1_score": avg("f1_score"),
    }

    print(f"\n📊 Aggregated Metrics:")
    for k, v in aggregated.items():
        print(f"  {k}: {v:.4f}")

    return aggregated

# ---- 2. Configure Strategy with Aggregation ----
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)

def main():
    # Configure and start Flower server
    fl.server.start_server(
        server_address="localhost:8081",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10)
    )

if __name__ == "__main__":
    main()
