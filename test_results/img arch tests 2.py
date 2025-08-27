import numpy as np
import matplotlib.pyplot as plt

with open("images archs 2.txt", "r") as f:
    lines = f.readlines()

def get_accs_and_losses(i):
    step_accs = [100*float(num) for num in lines[i][13:-2].split(',')]
    rand_accs = [100*float(num) for num in lines[i+1][20:-2].split(',')]
    losses = [float(num) for num in lines[i+2][9:-2].split(',')]
    return step_accs, rand_accs, losses

step_accs_1, rand_accs_1, losses_1 = get_accs_and_losses(4)
step_accs_3, rand_accs_3, losses_3 = get_accs_and_losses(17)
step_accs_5, rand_accs_5, losses_5 = get_accs_and_losses(30)
step_accs_7, rand_accs_7, losses_7 = get_accs_and_losses(43)
step_accs_9, rand_accs_9, losses_9 = get_accs_and_losses(56)

plt.plot(step_accs_1, label='1 hidden layer')
plt.plot(step_accs_3, label='3 hidden layers')
plt.plot(step_accs_5, label='5 hidden layers')
plt.plot(step_accs_7, label='7 hidden layers')
plt.plot(step_accs_9, label='9 hidden layers')

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title("Accuracy Curves")
plt.ylim(90,100)
plt.legend()
plt.grid(True)
plt.show()


plt.plot(losses_1, label='1 hidden layer')
plt.plot(losses_3, label='3 hidden layers')
plt.plot(losses_5, label='5 hidden layers')
plt.plot(losses_7, label='7 hidden layers')
plt.plot(losses_9, label='9 hidden layers')

plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("Loss Curves")
plt.ylim(0,0.1)
plt.legend()
plt.grid(True)
plt.show()
