import csv
import matplotlib.pyplot as plt
import os

def load_metrics(csv_path):
    rounds, acc, prec, rec, f1 = [], [], [], [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            acc.append(float(row["accuracy"]))
            prec.append(float(row["precision"]))
            rec.append(float(row["recall"]))
            f1.append(float(row["f1"]))
    return rounds, acc, prec, rec, f1

if __name__ == "__main__":
    csv_path = "runs/cm_fi/metrics.csv"
    rounds, acc, prec, rec, f1 = load_metrics(csv_path)

    out_dir = os.path.dirname(csv_path)

    # accuracy curve
    plt.figure()
    plt.plot(rounds, acc, marker="o")
    plt.title("Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150)
    plt.close()

    # f1 curve
    plt.figure()
    plt.plot(rounds, f1, marker="o")
    plt.title("F1 over Rounds")
    plt.xlabel("Round")
    plt.ylabel("F1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_curve.png"), dpi=150)
    plt.close()
