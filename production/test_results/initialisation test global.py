import jax.numpy as jnp
from collections import defaultdict

def mean_std(array):
    return jnp.mean(array), jnp.std(array)

with open("initialisation test global.txt", "r") as f:
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
while i < 2580:
    arch = lines[i][17:-2]
    i += 1
    sigma = lines[i][16:-2]
    i += 2
    distribution = lines[i][24:-3]
    print(distribution, sigma, arch)
    total_counts[(distribution, sigma, arch)] += 1
    i += 1
    line = lines[i].split(',')
    if len(line) == 3:
        acc, loss, rand_acc = line
        accuracies[(distribution, sigma, arch)].append(float(acc[10:-1]))
        losses[(distribution, sigma, arch)].append(float(loss[6:]))
        random_accuracies[(distribution, sigma, arch)].append(float(rand_acc[18:-2]))
        i += 4
    else:
        successful_counts[(distribution, sigma, arch)] += 1
        i += 2
        gate_usage[(distribution, sigma, arch)].append([int(node_count) for node_count in lines[i][1:-2].split(',')])
        print(lines[i])
        i += 3
        max_fan_ins[(distribution, sigma, arch)].append(int(lines[i][12:-1]))
        print(lines[i])
        i += 1
        average_fan_ins[(distribution, sigma, arch)].append(float(lines[i][16:-1]))
        print(lines[i])
        i += 5
        gate_pto[(distribution, sigma, arch)].append([int(node_count) for node_count in lines[i][1:-2].split(',')])
        print(lines[i])
        i += 3
        max_fan_pto[(distribution, sigma, arch)].append(int(lines[i][12:-1]))
        print(lines[i])
        i += 1
        average_fan_pto[(distribution, sigma, arch)].append(float(lines[i][16:-1]))
        print(lines[i])
        i += 3
        training_times[(distribution, sigma, arch)].append(float(lines[i][21:-9]))
        print(lines[i])
        i += 1
        accuracies[(distribution, sigma, arch)].append(100)
        random_accuracies[(distribution, sigma, arch)].append(100)

[print(item) for item in sorted([item for item in successful_counts.items()])]
print()

accuracies = {k:mean_std(jnp.array(v)) for k,v in accuracies.items()}
random_accuracies = {k:mean_std(jnp.array(v)) for k,v in random_accuracies.items()}
training_times = {k:mean_std(jnp.array(v)) for k,v in training_times.items()}
success_rates = {k:100*successful_counts[k]/total_counts[k] for k in total_counts.keys()}
max_fan_pto = {k:mean_std(jnp.array(v)) for k,v in max_fan_pto.items()}
average_fan_pto = {k:mean_std(jnp.array(v)) for k,v in average_fan_pto.items()}

print()
print(max([y for x,y in training_times.values()]))

import matplotlib.pyplot as plt

for dist in ("beta_sampler", "normal_sampler1", "normal_sampler2"):
    for arch in ("[160, 96]", "[192, 64]"):
        filt = lambda x: x[0] == dist and x[2] == arch

        plots = [("Step Accuracy (%)", "Step Accuracy by Standard Deviation", [v for k, v in filter(lambda item: filt(item[0]), accuracies.items())]),
                 ("Bernoulli Accuracy (%)", "Bernoulli Accuracy by Standard Deviation", [v for k, v in filter(lambda item: filt(item[0]), random_accuracies.items())]),
                 ("Training time (s)", "Training time by Standard Deviation", [v for k, v in sorted(filter(lambda item: filt(item[0]), training_times.items()))]),
                 ("Fan-in", "Max fan-in by Standard Deviation", [v for k, v in sorted(filter(lambda item: filt(item[0]), max_fan_pto.items()))]),
                 ("Fan-in", "Average fan-in by Standard Deviation", [v for k, v in sorted(filter(lambda item: filt(item[0]), average_fan_pto.items()))])]

        labels = [k[1] for k, v in filter(lambda item: filt(item[0]), accuracies.items())]
        training_time_labels = [k[1] for k, v in sorted(filter(lambda item: filt(item[0]), training_times.items()))]

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

            if len(means) != len(labels):
                plt.xticks(x, training_time_labels)
            else:
                plt.xticks(x, labels)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if title is None:
                plt.title(f"{ylabel} by {xlabel} ({dist}, {arch})")
            else:
                plt.title(f"{title} ({dist}, {arch})")
            plt.grid(axis='y', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.show()

        graph("Standard Deviation", "Success Rate (%)", [v for k, v in filter(lambda item: filt(item[0]), success_rates.items())])

        width = 0.35 
        x = jnp.arange(len(labels))
        acc_means, acc_stds = zip(*[v for k, v in filter(lambda item: filt(item[0]), accuracies.items())])
        bern_means, bern_stds = zip(*[v for k, v in filter(lambda item: filt(item[0]), random_accuracies.items())])
        plt.bar(x - width/2, acc_means, width, yerr=acc_stds, label='Step Accuracy (%)', capsize=5, edgecolor='black')
        plt.bar(x + width/2, bern_means, width, yerr=bern_stds, label='Bernoulli Accuracy (%)', capsize=5, edgecolor='black')
        plt.legend(loc="lower left")
        plt.xticks(x, labels)
        plt.xlabel("Hidden layers")
        plt.ylabel("Accuracy (%)")
        plt.ylim(80,100)
        plt.title(f"Accuracy with Step vs Bernoulli by Architecture ({dist}, {arch})")
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        for ylabel, title, data in plots:
            means, stds = zip(*data)
            graph("Standard Deviation", ylabel, means, stds, title)
