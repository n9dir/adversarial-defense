import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

baseline = {
    "Clean": 99.05,
    "FGSM": 10.97,
    "PGD": 0.00,
}

defended = {
    "Clean": 98.91,
    "FGSM": 93.99,
    "PGD": 89.94,
}

labels = list(baseline.keys())
baseline_values = list(baseline.values())
defended_values = list(defended.values())
x = range(len(labels))
baseline_bars = [max(value, 0.8) for value in baseline_values]
defended_bars = [max(value, 0.8) for value in defended_values]

plt.figure(figsize=(8, 5))
baseline_plot = plt.bar([i - 0.2 for i in x], baseline_bars, width=0.4, label="Baseline Model")
defended_plot = plt.bar([i + 0.2 for i in x], defended_bars, width=0.4, label="Adversarially Trained Model")
plt.xticks(x, labels)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.title("Model Performance Under Adversarial Attacks")
plt.legend()

for bars, values in ((baseline_plot, baseline_values), (defended_plot, defended_values)):
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.savefig("results_comparison.png")
