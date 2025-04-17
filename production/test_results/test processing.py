import jax.numpy as jnp

results = ["8 NAND gates.txt", "16 NAND gates.txt", "32 NAND gates.txt"]

max_fan_in_bads = []
max_fan_in_goods = []
average_fan_in_bads = []
average_fan_in_goods = []
max_ptos = []
average_ptos = []
timess = []
success_rates = []

def mean_std(array):
    return jnp.mean(array), jnp.std(array)

for result in results:

    with open(result, "r") as f:
        lines = f.readlines()

    print(result)
    timeout = 120
    # each test is 18 lines
    # max fan-in inefficient would be on line 7 (index 6)
    # average fan-in inefficient would be on line 8 (index 7)
    # max fan-in efficient would be in line 15 (index 14)
    # average fan-in efficient would be on line 16 (index 15)
    # timeout would be on line 18 (index 17, from index 21 of the line)

    max_fan_in_bad = []
    max_fan_in_good = []
    average_fan_in_bad = []
    average_fan_in_good = []
    times = []
    for i in range(20):
        time = float(lines[18*i + 17][21:25])
        if time < timeout:
            max_fan_in_bad.append(int(lines[18*i + 6][12:]))
            average_fan_in_bad.append(float(lines[18*i + 7][16:]))
            max_fan_in_good.append(int(lines[18*i + 14][12:]))
            average_fan_in_good.append(float(lines[18*i + 15][16:]))
            times.append(time)

    max_fan_in_bad = jnp.array(max_fan_in_bad)
    max_fan_in_good = jnp.array(max_fan_in_good)
    average_fan_in_bad = jnp.array(average_fan_in_bad)
    average_fan_in_good = jnp.array(average_fan_in_good)
    max_fan_pto = max_fan_in_bad - max_fan_in_good
    average_fan_pto = average_fan_in_bad - average_fan_in_good
    times = jnp.array(times)

    max_fan_in_bads.append(mean_std(max_fan_in_bad))
    max_fan_in_goods.append(mean_std(max_fan_in_good))
    average_fan_in_bads.append(mean_std(average_fan_in_bad))
    average_fan_in_goods.append(mean_std(average_fan_in_good))
    max_ptos.append(mean_std(max_fan_pto))
    average_ptos.append(mean_std(average_fan_pto))
    timess.append(mean_std(times))
    success_rates.append(len(times) * 5)


    
import matplotlib.pyplot as plt

plots = [("Max fan-in", max_fan_in_bads),
         ("Average fan-in", average_fan_in_bads),
         ("Max fan-in with PTO", max_fan_in_goods),
         ("Average fan-in with PTO", average_fan_in_goods),
         ("Training time (s)", timess),
         ("PTO max", max_ptos),
         ("PTO average", average_ptos)]

labels = ["8", "16", "32"]

def graph(xlabel, ylabel, means, stds=None, title=None):
    x = range(len(means))
    print(ylabel, means, stds)
    
    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=10,
        width=0.6,
        edgecolor='black',
    )

    plt.xticks(x, labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        plt.title(f"{ylabel} by {xlabel}")
    else:
        plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

means, stds = zip(*max_ptos)
graph("No. NAND gates in hidden layer", "Δ Max fan-in", means, stds)

for title, data in plots:
    means, stds = zip(*data)
    graph("No. NAND gates in hidden layer", title, means, stds)

graph("No. NAND gates in hidden layer", "Success rate (%)", success_rates)

