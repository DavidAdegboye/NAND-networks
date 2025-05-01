import numpy as np
import matplotlib.pyplot as plt

with open("images archs 1.txt", "r") as f:
    lines = f.readlines()

def get_accs_and_losses(i):
    step_accs = [100*float(num) for num in lines[i][13:-2].split(',')]
    rand_accs = [100*float(num) for num in lines[i+1][20:-2].split(',')]
    losses = [float(num) for num in lines[i+2][9:-2].split(',')]
    return step_accs, rand_accs, losses

step_accs_1, rand_accs_1, losses_1 = get_accs_and_losses(4)
step_accs_2, rand_accs_2, losses_2 = get_accs_and_losses(17)
step_accs_3, rand_accs_3, losses_3 = get_accs_and_losses(30)
step_accs_4, rand_accs_4, losses_4 = get_accs_and_losses(43)
step_accs_5, rand_accs_5, losses_5 = get_accs_and_losses(56)
step_accs_6, rand_accs_6, losses_6 = get_accs_and_losses(69)
step_accs_7, rand_accs_7, losses_7 = get_accs_and_losses(82)
step_accs_8, rand_accs_8, losses_8 = get_accs_and_losses(95)
step_accs_9, rand_accs_9, losses_9 = get_accs_and_losses(108)

plt.plot(step_accs_1, label='1 hidden layer')
plt.plot(step_accs_2, label='2 hidden layers')
plt.plot(step_accs_3, label='3 hidden layers')
plt.plot(step_accs_4, label='4 hidden layers')
plt.plot(step_accs_5, label='5 hidden layers')
plt.plot(step_accs_6, label='6 hidden layers')
plt.plot(step_accs_7, label='7 hidden layers')
plt.plot(step_accs_8, label='8 hidden layers')
plt.plot(step_accs_9, label='9 hidden layers')

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title("Accuracy Curves, 1 hr timeout")
plt.ylim(90,100)
plt.legend()
plt.grid(True)
plt.show()


plt.plot(losses_1, label='1 hidden layer')
plt.plot(losses_2, label='2 hidden layers')
plt.plot(losses_3, label='3 hidden layers')
plt.plot(losses_4, label='4 hidden layers')
plt.plot(losses_5, label='5 hidden layers')
plt.plot(losses_6, label='6 hidden layers')
plt.plot(losses_7, label='7 hidden layers')
plt.plot(losses_8, label='8 hidden layers')
plt.plot(losses_9, label='9 hidden layers')

plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title("Loss Curves, 1 hr timeout")
plt.ylim(0,0.1)
plt.legend()
plt.grid(True)
plt.show()

with open("images archs 2.txt", "r") as f:
    lines = f.readlines()

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
plt.title("Accuracy Curves, 2 hr timeout")
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
plt.title("Loss Curves, 2 hr timeout")
plt.ylim(0,0.1)
plt.legend()
plt.grid(True)
plt.show()
