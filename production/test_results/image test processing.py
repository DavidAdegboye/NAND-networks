import numpy as np
import matplotlib.pyplot as plt

with open("images accs and losses.txt", "r") as f:
    lines = f.readlines()

def get_accs_and_losses(i):
    step_accs = [100*float(num) for num in lines[i][13:-2].split(',')]
    rand_accs = [100*float(num) for num in lines[i+1][20:-2].split(',')]
    losses = [float(num) for num in lines[i+2][9:-2].split(',')]
    return step_accs, rand_accs, losses

t_start = 0
t_end = 900

step_accs, rand_accs, losses = get_accs_and_losses(4)
step_accs_comp, rand_accs_comp, losses_comp = get_accs_and_losses(15)

t1 = np.linspace(t_start, t_end, len(step_accs))
t2 = np.linspace(t_start, t_end, len(step_accs_comp))

plt.plot(t1, step_accs, label='Baseline')
# plt.plot(t1, rand_accs, label='Bernoulli Accuracy Baseline')
plt.plot(t2, step_accs_comp, label='Pooling')
# plt.plot(t2, rand_accs_comp, label='Bernoulli Accuracy Pooling')

plt.xlabel('Time (s)')
plt.ylabel('Accuracy (%)')
plt.title("Accuracy Curves")
plt.ylim(90,100)
plt.legend()
plt.grid(True)
plt.show()

step_accs_comp, rand_accs_comp, losses_comp = get_accs_and_losses(26)
t2 = np.linspace(t_start, t_end, len(step_accs_comp))
step_accs_comp2, rand_accs_comp2, losses_comp2 = get_accs_and_losses(48)
t3 = np.linspace(t_start, t_end, len(step_accs_comp2))

plt.plot(t1, step_accs, label='min_gates=[0,0,0]')
# plt.plot(t1, rand_accs, label='Bernoulli Accuracy Baseline')
plt.plot(t2, step_accs_comp, label='min_gates=[784, 1024, 5]')
# plt.plot(t2, rand_accs_comp, label='Bernoulli Accuracy Pooling')
plt.plot(t3, step_accs_comp2, label='min_gates=[1568, 2048, 10]')
# plt.plot(t3, rand_accs_comp2, label='Bernoulli Accuracy Pooling')

plt.xlabel('Time (s)')
plt.ylabel('Accuracy (%)')
plt.title("Accuracy Curves")
plt.ylim(90,100)
plt.legend()
plt.grid(True)
plt.show()

step_accs_comp, rand_accs_comp, losses_comp = get_accs_and_losses(70)
t2 = np.linspace(t_start, t_end, len(step_accs_comp))

plt.plot(t1, step_accs, label='hidden_layers=[2048]')
# plt.plot(t1, rand_accs, label='Bernoulli Accuracy Baseline')
plt.plot(t2, step_accs_comp, label='hidden_layers=[1024, 784, 512, 256]')
# plt.plot(t2, rand_accs_comp, label='Bernoulli Accuracy Pooling')

plt.xlabel('Time (s)')
plt.ylabel('Accuracy (%)')
plt.title("Accuracy Curves")
plt.ylim(90,100)
plt.legend()
plt.grid(True)
plt.show()
