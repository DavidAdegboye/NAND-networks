import jax.numpy as jnp
from collections import defaultdict

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


with open("16-input test.txt", "r") as f:
    lines = f.readlines()

# each test is 7 lines
# continuous_penalty_coeff would be at index 0 (28:)
# architecture would be at index 1 (16:)
# acc and loss at index 3 (split on comma)
# max fan-in at index 5 (12:)

accuracies = defaultdict(list)
random_accuracies = defaultdict(list)
losses = defaultdict(list)
fan_ins = defaultdict(list)

for i in range(56):
    cpc = float(lines[7*i][28:])
    layers = (len(lines[7*i+1][16:])-1)//4
    acc, loss, rand_acc = lines[7*i+3].split(',')
    accuracies[(cpc, layers)].append(float(acc[10:-1]))
    losses[(cpc, layers)].append(float(loss[6:]))
    random_accuracies[(cpc, layers)].append(float(rand_acc[18:-2]))
    fan_ins[(cpc, layers)].append(int(float(lines[7*i+5][12:-1])))

accuracies_1 = []
losses_1 = []
random_accuracies_1 = []
fan_ins_1 = []
accuracies_2 = []
losses_2 = []
random_accuracies_2 = []
fan_ins_2 = []

for cpc,layers in accuracies.keys():
    if layers == 1:
        accuracies_1.append(mean_std(jnp.array(accuracies[(cpc,layers)])))
        losses_1.append(mean_std(jnp.array(losses[(cpc,layers)])))
        random_accuracies_1.append(mean_std(jnp.array(random_accuracies[(cpc,layers)])))
        fan_ins_1.append(mean_std(jnp.array(fan_ins[(cpc,layers)])))
    else:
        accuracies_2.append(mean_std(jnp.array(accuracies[(cpc,layers)])))
        losses_2.append(mean_std(jnp.array(losses[(cpc,layers)])))
        random_accuracies_2.append(mean_std(jnp.array(random_accuracies[(cpc,layers)])))
        fan_ins_2.append(mean_std(jnp.array(fan_ins[(cpc,layers)])))
    
import matplotlib.pyplot as plt

plots = [("Accuracy (%)", "Accuracy by CPC for 1 hidden layer", accuracies_1),
         ("Loss", "Loss by CPC for 1 hidden layer", losses_1),
         ("Bernoulli Accuracy (%)", "Bernoulli Accuracy by CPC for 1 hidden layer", random_accuracies_1),
         ("Max fan-in", "Max fan-in by CPC for 1 hidden layer", fan_ins_1),
         ("Accuracy (%)", "Accuracy by CPC for 2 hidden layers", accuracies_2),
         ("Loss", "Loss by CPC for 2 hidden layers", losses_2),
         ("Bernoulli Accuracy (%)", "Bernoulli Accuracy by CPC for 2 hidden layers", random_accuracies_2),
         ("Max fan-in", "Max fan-in by CPC for 2 hidden layers", fan_ins_2)]

labels = ["0", "0.003", "0.01", "0.05", "0.1", "0.5", "1"]

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

for ylabel, title, data in plots:
    means, stds = zip(*data)
    graph("Continuous Penalty Coefficient (CPC)", ylabel, means, stds, title)


