import jax.numpy as jnp
from collections import defaultdict

def mean_std(array):
    return jnp.mean(array), jnp.std(array)

with open("initialisation test.txt", "r") as f:
    lines = f.readlines()

# first architecture (17:)
# then max_fan (skip)
# if failed, next is accuracy, loss, bernoulli accuracy
# gate usage (12:)
# max fan (12:)

# otherwise skip this and next
# gate usage without PTO
# skip this and next
# max fan
# average fan
# skip this and next two
# gate usage with PTO
# skip this and next
# max fan
# average fan
# skip
# training time

# by architecture, we want success rate, training time on successful runs,
# (max fan-in, average fan-in, gate usage) before and after PTO, and readouts for failed tests

total_counts = defaultdict(int)
successful_counts = defaultdict(int)
training_times = defaultdict(list)
max_fan_ins = defaultdict(list)
average_fan_ins = defaultdict(list)
gate_usage = defaultdict(list)
max_fan_pto = defaultdict(list)
average_fan_pto = defaultdict(list)
gate_pto = defaultdict(list)
accuracies = defaultdict(list)
losses = defaultdict(list)
random_accuracies = defaultdict(list)

success = False
i=0
while i < 498:
    arch = lines[i][17:-2]
    i += 1
    sigma = lines[i][16:-2]
    i += 2
    distribution = lines[i][24:-3]
    print(arch, sigma, distribution)
    total_counts[(arch, sigma, distribution)] += 1
    i += 2
    line = lines[i].split(',')
    if len(line) == 3:
        acc, loss, rand_acc = line
        accuracies[(arch, sigma, distribution)].append(float(acc[10:-1]))
        losses[(arch, sigma, distribution)].append(float(loss[6:]))
        random_accuracies[(arch, sigma, distribution)].append(float(rand_acc[18:-2]))
        i += 4
    else:
        successful_counts[(arch, sigma, distribution)] += 1
        i += 2
        gate_usage[(arch, sigma, distribution)].append([int(node_count) for node_count in lines[i][1:-2].split(',')])
        print(lines[i])
        i += 3
        max_fan_ins[(arch, sigma, distribution)].append(int(lines[i][12:-1]))
        print(lines[i])
        i += 1
        average_fan_ins[(arch, sigma, distribution)].append(float(lines[i][16:-1]))
        print(lines[i])
        i += 5
        gate_pto[(arch, sigma, distribution)].append([int(node_count) for node_count in lines[i][1:-2].split(',')])
        print(lines[i])
        i += 3
        max_fan_pto[(arch, sigma, distribution)].append(int(lines[i][12:-1]))
        print(lines[i])
        i += 1
        average_fan_pto[(arch, sigma, distribution)].append(float(lines[i][16:-1]))
        print(lines[i])
        i += 3
        training_times[(arch, sigma, distribution)].append(float(lines[i][21:-9]))
        print(lines[i])
        i += 1
        accuracies[(arch, sigma, distribution)].append(100)
        random_accuracies[(arch, sigma, distribution)].append(100)

print(successful_counts)
print()
print(total_counts)

accuracies = {k:mean_std(jnp.array(v)) for k,v in accuracies.items()}
random_accuracies = {k:mean_std(jnp.array(v)) for k,v in random_accuracies.items()}
training_times = {k:mean_std(jnp.array(v)) for k,v in training_times.items()}
print()
print(accuracies)
print()
print(random_accuracies)

"""
import matplotlib.pyplot as plt

plots = [("Accuracy (%)", "Accuracy by Architecture", accuracies.values()),
         ("Bernoulli Accuracy (%)", "Bernoulli Accuracy by Architecture", random_accuracies.values()),
         ("Training time (s)", "Training time by Architecture", training_times.values()),
         ("Fan-in", "Max fan-in by Architecture", max_fan_pto.values()),
         ("Fan-in", "Average fan-in by Architecture", average_fan_pto.values())]

labels = accuracies.keys()

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
    graph("Architecture", ylabel, means, stds, title)

width = 0.35 
x = jnp.arange(len(labels))
acc_means, acc_stds = zip(*accuracies.values())
bern_means, bern_stds = zip(*random_accuracies.values())
plt.bar(x - width/2, acc_means, width, yerr=acc_stds, label='Step Accuracy (%)', capsize=5, edgecolor='black')
plt.bar(x + width/2, bern_means, width, yerr=bern_stds, label='Bernoulli Accuracy (%)', capsize=5, edgecolor='black')
plt.legend(loc="lower left")
plt.xticks(x, labels)
plt.xlabel("Hidden layers")
plt.ylabel("Accuracy (%)")
plt.ylim(99,100)
plt.title("Accuracy with Step vs Bernoulli by Architecture")
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
"""
