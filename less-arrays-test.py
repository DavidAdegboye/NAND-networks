import jax
import jax.numpy as jnp
import optax
import random
import math
import itertools
from typing import List, Tuple, Set, Union
import time

# in some sense a neuron is a list of layers also, which 
# can cause some logical bugs. For example, f is calculating
# values over a layer

# want neuron to be an array of arrays
Network = List[jnp.ndarray]
NeuronShape = List[int]
LayerShape = List[NeuronShape]
NetworkShape = List[LayerShape]
##jax.config.update("jax_traceback_filtering", "off")

print(jax.devices())
bits = int(input("Input bits:\n"))
def denary_to_binary_array(number: jnp.ndarray, bits: int=bits*2) -> jnp.ndarray:
    return jnp.array([(jnp.right_shift(number, bits - 1 - i) & 1) for i in range(bits)], dtype=jnp.int32)

def get_output(number: jnp.ndarray) -> jnp.ndarray:
    return denary_to_binary_array(number//(2**bits) + number%(2**bits), bits=bits+1)

inputs = jax.vmap(denary_to_binary_array)(jnp.arange(2**(bits*2)))
output = jax.vmap(get_output)(jnp.arange(2**(bits*2)))

# arch = [4,19,15,10,7,5,3]
# arch = [3,4,3,3,3,2]
# arch = [2,1,2,2]
ins = bits*2
outs = bits+1

def add_second_layers(input: jnp.ndarray, min_fan: int, max_fan: int) -> jnp.ndarray:
    # giving the network the second layer for free. Can hypothetically do this n times, although its cost grows exponentially.
    output = list(input)
    unchanged = output.copy()
    for k in range(min_fan, max_fan+1):
        for comb in itertools.combinations(unchanged, k):
            output.append(1-jnp.prod(jnp.array(comb)))
    return jnp.array(output)

def help_adder(input: jnp.ndarray, nots: bool) -> jnp.ndarray:
    output = list(input)
    for i in range(bits):
        output.append(1-output[i]*output[i+bits])
        # so for example ABC+DEF, we're adding A NAND D, B NAND E and C NAND F
        if nots:
            output.append(1-output[i]*output[i+3*bits])
            output.append(1-output[i+bits]*output[i+2*bits])
            output.append(1-output[i+2*bits]*output[i+3*bits])
    return jnp.array(output)

extra_layers = []
true_arch = []
add_extra = input("Add extra layer? Yes(y) or no(n)\n")
while add_extra == 'y':
    min_fan = int(input("Min fan-in of this layer:\n"))
    min_fan = max(min_fan, 1)
    max_fan = int(input("Max fan-in of this layer:\n"))
    max_fan = min(max_fan, inputs.shape[1])
    extra_layers.append((min_fan, max_fan))
    old_ins = inputs.shape[1]
    inputs = jax.vmap(add_second_layers, in_axes=(0, None, None))(inputs, min_fan, max_fan)
    mask = jnp.sum(inputs, axis=0) < 2**ins
    inputs = inputs[:, mask]
    print(inputs)
    new_ins = inputs.shape[1]
    true_arch.append(new_ins - old_ins)
    add_extra = input("Add another extra layer? Yes(y) or no(n)\n")

add_adder_help = input("Add extra help for learning an adder? Yes(y) or no(n)\n")
if add_adder_help == 'y':
    with_nots = input("Did you add a complement layer? Yes(y) or no(n)\n")
    old_ins = inputs.shape[1]
    inputs = jax.vmap(help_adder, in_axes=(0, None))(inputs, with_nots=='y')
    new_ins = inputs.shape[1]
    true_arch.append(new_ins - old_ins)

print(true_arch)
new_ins = inputs.shape[1]

global_weights = input("Global weights(g) or local(l)?\n")

start_i = int(input("Input starting index (recommend 1 for local, 10 for global):\n"))
end_i = int(input("Input ending index (max 24, recommend 8 for local, 11 for global):\n"))
taper_q = input("Taper (t), flat(f), custom(c) or linear(l)?\n")
if taper_q == 't':
    layer2 = 2**ins - 1
    taper = float(input("Input taper ratio:\n"))
    next_layer = round(layer2 * taper)
    arch = [new_ins]
    while next_layer > outs:
        arch.append(next_layer)
        next_layer = min(next_layer-1, round(next_layer*taper))
    if arch[-1] != outs:
        arch.append(outs)
elif taper_q == 'c':
    arch = [new_ins]
    next_hidden = int(input("How many neurons in the next layer? Input 0 to say no more hidden layers\n"))
    while next_hidden:
        arch += [next_hidden]
        next_hidden = int(input("How many neurons in the next layer? Input 0 to say no more hidden layers\n"))
    arch += [outs]
elif taper_q == 'f':
    width = int(input("Arch Width:\n"))
    hidden = int(input("No. hidden layers:\n"))
    wide = input("Add width(w) or depth(d)?\n")
    arch = [new_ins] + [width] * hidden + [outs]
else:
    starting_width = int(input("Starting Width:\n"))
    hidden = int(input("No. hidden layers:\n"))
    wide = input("Add width(w) or depth(d)?\n")
    diff = starting_width - outs
    layer_diff = diff/hidden
    arch = [new_ins] + [round(starting_width-i*layer_diff) for i in range(hidden)] + [outs]

l2_coeff = float(input("l2 coefficient:\n"))

@jax.jit
def f(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    # x would be all of the inputs coming in from a certain layer
    # w would be all of the weights for inputs to that layer to a given NAND gate
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w))

@jax.jit
def f_disc(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return jnp.prod(jnp.where(w>0, x, 1)) 

@jax.jit
def forward(weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    # the forward pass for an arbitrary neuron. 1 - the product of all the fs
    # to use vmap, I include some padding that doesn't affect the value.
    # x=1, w=0, since f(1,0)=1, so it wouldn't affect the result
    # after the product.
    return 1 - jnp.prod(jax.vmap(f)(xs, weights))

@jax.jit
def forward_disc(weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    return 1 - jnp.prod(jax.vmap(f_disc)(xs, weights))

def get_used(used: List[int], arch: List[int]):
    output = []
    current = 0
    layer_i = 0
    current_l = 0
    if true_arch:
        eff_arch = arch[1:].copy()
        eff_arch = [ins] + true_arch + eff_arch
    else:
        eff_arch = arch.copy()
    print(eff_arch)
    current_h = eff_arch[0]
    for node in used:
        if current_l <= node < current_h:
            current += 1
        else:
            while not(current_l <= node < current_h):
                output.append(current)
                current = 0
                layer_i += 1
                current_l = current_h
                current_h = current_l + eff_arch[layer_i]
            current += 1
    output.append(current)
    output += [0] * (len(output) - len(eff_arch) - 1)
    output.append(outs)
    return output

def output_circuit(neurons: Network, verbose=False) -> List[str]:
    connecteds: List[List[int]] = [[] for _ in range(ins)]
    if extra_layers:
        circuits = [chr(ord('A')+i) for i in range(ins)]
        for layer in extra_layers:
            print(layer)
            if layer[0] == 1:
                extras = ["¬"+circ for circ in circuits]
                connecteds += [[i] for i in range(len(connecteds))]
            else:
                extras = []
            for k in range(max(2, layer[0]), layer[1]+1):
                for comb in itertools.combinations(circuits, k):
                    add = True
                    for gate in comb:
                        if "¬"+gate in comb:
                            add = False
                    if add:
                        extras.append("¬("+".".join(comb)+")")
                        new_con = []
                        for gate in comb:
                            new_con.append(circuits.index(gate))
                        connecteds.append(new_con)
            circuits += extras
        if add_adder_help == 'y':
            for i in range(bits):
                circuits.append("¬(" + chr(ord('A')+i) + "." + chr(ord('A')+i+bits) + ")")
                connecteds.append([i,i+bits])
                # so for example ABC+DEF, we're adding A NAND D, B NAND E and C NAND F
                if with_nots == 'y':
                    circuits.append("¬(" + chr(ord('A')+i) + ".¬" + chr(ord('A')+i+bits) + ")")
                    circuits.append("¬(¬" + chr(ord('A')+i) + "." + chr(ord('A')+i+bits) + ")")
                    circuits.append("¬(¬" + chr(ord('A')+i) + ".¬" + chr(ord('A')+i+bits) + ")")
                    connecteds.append([i,i+3*bits])
                    connecteds.append([i+bits,i+2*bits])
                    connecteds.append([i+2*bits,i+3*bits])
    else:
        circuits = [chr(ord('A')+i) for i in range(arch[0])]
    gates:List[List[List[Union[str,Tuple[int,int]]]]] = [[[] for _ in range(arch[0])]]
    c2i = dict([(x,i) for i,x in enumerate(circuits)])
    indices = dict([(i,i) for i in range(arch[0])])
    index2gate = dict([(i, (0,i)) for i in range(arch[0])])
    empties = []
    added = arch[0] - 1
    used = set()
    for layer_i in range(i_1):
        gates.append([])
        gate_i1 = layer_i+1
        gate_i2 = 0
        for neuron_i in range(len(shapes[layer_i])):
            i = 0
            connected: Set[Tuple[int, str]] = set()
            for inner_layer_i in range(len(shapes[layer_i][neuron_i])):
                for weight_i in range(shapes[layer_i][neuron_i][inner_layer_i]):
                    if neurons[layer_i][neuron_i,inner_layer_i,weight_i] > 0 and indices[i] not in empties:
                        connected.add((indices[i], circuits[indices[i]]))
                    i += 1
            added += 1
            connected = sorted(list(connected))
            connecteds.append([node[0] for node in connected])
            i = len(connecteds)-1
            # print(i, connecteds[i])
            if not connected:
                empties.append(added)
                indices[added] = added
                circuits.append('_')
            else:
                if len(connected) == 1:
                    node = '¬' + connected[0][1]
                    if len(node) > 2:
                        if node[:3] == "¬¬¬":
                            node = node[2:]
                else:
                    node = '¬(' + '.'.join([element[1] for element in connected]) + ')'
                if node in c2i.keys():
                    if layer_i == i_1-1:
                        circuits.append(node)
                        gates[-1].append(["=", index2gate[c2i[node]]])
                        index2gate[added] = (gate_i1, gate_i2)
                        gate_i2 += 1
                        for prev_node in connected:
                            used.add(prev_node[0])
                    else:
                        circuits.append('_')
                    indices[added] = c2i[node]
                else:
                    circuits.append(node)                    
                    c2i[node] = added
                    indices[added] = added
                    gates[-1].append([index2gate[element[0]] for element in connected])
                    index2gate[added] = (gate_i1, gate_i2)
                    gate_i2 += 1
                    if layer_i == i_1-1:
                        for prev_node in connected:
                            used.add(prev_node[0])
    queue = list(used)
    nodes = []
    while len(queue):
        node_i = queue.pop(0)
        nodes.append(node_i)
        for node_2 in connecteds[node_i]:
            if node_2 not in used:
                queue.append(node_2)
                used.add(node_2)
    if verbose:
        print(c2i)
        print(indices)
        print(circuits)
        print(gates)
    # print(nodes)
    used: List[int] = sorted(list(used))
    print("used:\n", get_used(used, arch), "\nout of:\n", arch)
    return circuits[-arch[-1]:]

@jax.jit
def feed_forward(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
    for layer_i in range(i_1-1):
        next = jax.vmap(forward, in_axes=(0, None))(neurons[layer_i], xs)
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    return jax.vmap(forward, in_axes=(0, None))(neurons[i_1-1], xs)[:len(output[0])]

@jax.jit
def feed_forward_disc(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
    for layer_i in range(i_1-1):
        next = jax.vmap(forward_disc, in_axes=(0, None))(neurons[layer_i], xs)
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    return jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:len(output[0])]

def feed_forward_disc_print(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
    jax.debug.print("inputs:{}", inputs)
    jax.debug.print("{}", xs)
    for layer_i in range(i_1-1):
        next = jax.vmap(forward_disc, in_axes=(0, None))(neurons[layer_i], xs)
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
        jax.debug.print("{}", xs)
    jax.debug.print("{}", jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:len(output[0])])
    return jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:len(output[0])]

def get_weights(layer: int, arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    global key
    weights = jnp.ones((layer,i_4)) * -jnp.inf
    # layer lists, each with arch[i] elements
    # so this is a 2D list of floats
    # or a 1D list of jnp arrays
    if global_weights == 'g':
        n = global_n
    else:
        n = sum(arch[:layer])
    mu = -jnp.log(n-1)/k
    for i in range(layer):
        inner_layer = sigma * jax.random.normal(jax.random.key(key), (arch[i])) + mu
        weights = weights.at[i].set(jnp.pad(inner_layer, (0, i_4-arch[i]), mode="constant", constant_values=-jnp.inf))
        key = random.randint(0, 10000)
    return weights

def initialise(arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> Network:
    neurons = []
    for i1 in range(1, len(arch)):
        layer = jnp.ones((arch[i1], i1, i_4))
        for i2 in range(arch[i1]):
            layer = layer.at[i2].set(get_weights(i1, arch, sigma, k))
        neurons.append(layer)
    return neurons

def get_shapes(arch: List[int]) -> Tuple[NetworkShape, int]:
    shapes: NetworkShape = []
    total = 0
    for layer in range(1, len(arch)):
        shapes.append([])
        for _ in range(arch[layer]):
            shapes[-1].append(arch[:layer].copy())
            total += sum(arch[:layer])
    return shapes, total

@jax.jit
def get_l2(neurons: Network) -> float:
    s = 0
    for layer in neurons:
        s += jnp.sum(1-jax.nn.sigmoid(jnp.absolute(layer)))
    return s/total

epsilon = 1e-7
@jax.jit
def loss(neurons : Network) -> float:
    pred = jax.vmap(feed_forward, in_axes=(0, None))(inputs, neurons)
    pred = jnp.clip(pred, epsilon, 1-epsilon)
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    l2 = get_l2(neurons)
    return l1 + l2_coeff * l2

@jax.jit
def test(neurons: Network) -> jnp.ndarray:
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neurons)
    return jnp.all(pred==output)

@jax.jit
def acc(neurons: Network) -> jnp.ndarray:
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neurons)
    return jnp.sum(pred==output)/((2**(ins))*(outs))

## TODO : add jit, wrap loss function in jax value and grad, vmap, jax.debug.print
## every n iterations, discretize the circuit, test and potentially stop if 100% accurate
## try on adders, other 2 bit logic gates.

def solver_init(neurons: jnp.ndarray, opt_state, pass_through=False):
    return jax.lax.cond(pass_through, lambda _:opt_state, lambda _:solver.init(neurons), operand=None)

print("Learning:\n", output, "\nwith arch:", arch)
key = random.randint(0, 10000)

i_1 = len(arch) - 1
i_2 = max(arch[1:])
i_3 = i_1
i_4 = max(arch)
shapes, total = get_shapes(arch)
global_n = sum([arch[layer]*sum(arch[:layer]) for layer in range(1, len(arch))])/sum(arch)
print(global_n)
all_sigmas = jnp.array([0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
all_ks = jnp.array([1.0, 0.99, 0.98, 0.97, 0.955, 0.94, 0.92, 0.91, 0.9, 0.85, 0.75, 0.65, 0.5, 0.39, 0.32, 0.27, 0.23, 0.205, 0.18, 0.17, 0.155, 0.14, 0.13, 0.12, 0.11])
sigmas = all_sigmas[start_i:end_i]
ks = all_ks[start_i:end_i]
n = sigmas.shape[0]
start_time = time.time()
neuronss = [initialise(arch, sigmas[i], ks[i]) for i in range(n)]
old_losses = [loss(neuronss[index]) for index in range(n)]
solver = optax.adam(learning_rate=0.003)
opt_states = [solver.init(neurons) for neurons in neuronss]
restart_mask = [False] * n
init_time = time.time()
print("Took", init_time-start_time, "seconds to initialise.")
# print(opt_states[0])
cont = True
print("Losses:")
print([round(float(old_loss),5) for old_loss in old_losses])
print("Accuracies:")
print([round(float(acc(neurons)),5) for neurons in neuronss])
grad = jax.jit(jax.grad(loss))
iters = 0
popped = 0

while cont:
    iters += 1
    for _ in range(10):
        for i in range(n-popped):
            updates, opt_states[i] = solver.update(grad(neuronss[i]), opt_states[i], neuronss[i])
            neuronss[i] = optax.apply_updates(neuronss[i], updates)
    for index, neurons in enumerate(neuronss):
        if test(neurons):
            print(index)
            # jax.debug.print("neurons:{}", neurons)
            # print(jax.vmap(feed_forward_disc_print, in_axes=(0, None))(inputs, neuronss[index]))
            final_index = index
            cont = False
    if cont:
        new_losses = [loss(neuronss[index]) for index in range(n-popped)]
        for i in range(n-popped):
            if old_losses[i] <= new_losses[i]:
                restart_mask[i] = True
                print("Restarting sigma value", sigmas[i], "with new random weights, stuck in local minima", new_losses[i], old_losses[i])
                # print(output_circuit(neuronss[i]))
                # print(jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neuronss[i]))
                neuronss[i] = initialise(arch, sigmas[i], ks[i])
                old_losses[i] = loss(neuronss[i]) 
                opt_states[i] = solver.init(neuronss[i])
            else:
                old_losses[i] = new_losses[i]
        if all(restart_mask):
            popped = 0
            restart_mask = [False] * n
            if taper_q == 't':
                taper = min(0.99, taper*1.1)
                next_layer = round(layer2*taper)
                arch = [new_ins]
                while next_layer > outs:
                    arch.append(next_layer)
                    next_layer = min(next_layer - 1, round(next_layer * taper))
                if arch[-1] != outs:
                    arch.append(outs)
            elif taper_q == 'c':
                arch += [outs]
            elif taper_q == 'f':
                if wide == 'w':
                    width = round(width*1.1)
                else:
                    hidden += 1
                arch = [new_ins] + [width] * hidden + [outs]
            else:
                if wide == 'w':
                    starting_width = round(starting_width*1.1)
                    diff = starting_width-outs
                    layer_diff = diff/hidden
                else:
                    hidden += 1
                    layer_diff = diff/hidden
                arch = [new_ins] + [round(starting_width-i*layer_diff) for i in range(hidden)] + [outs]
            end_time = time.time()
            print("Took", end_time-init_time, "seconds to get stuck.")
            i_1 = len(arch) - 1
            i_2 = max(arch[1:])
            i_3 = i_1
            i_4 = max(arch)
            shapes, total = get_shapes(arch)
            global_n = sum([arch[layer]*sum(arch[:layer]) for layer in range(1, len(arch))])/sum(arch)
            sigmas = all_sigmas[start_i:end_i]
            ks = all_ks[start_i:end_i]
            neuronss = [initialise(arch, sigmas[i], ks[i]) for i in range(n)]
            new_losses = [loss(neuronss[index]) for index in range(n)]
            old_losses = new_losses.copy()
            opt_states = [solver.init(neurons) for neurons in neuronss]
            init_time = time.time()
            print("New arch:", arch)
            print("Losses:")
            print([round(float(new_loss),5) for new_loss in new_losses])
            print("Accuracies:")
            print([round(float(acc(neuronss[index])),5) for index in range(n)])
            print("Took", init_time-end_time, "seconds to initialise.")
        if iters == 10:
            accuracies = [float(acc(neuronss[index])) for index in range(n-popped)]
            for i in range(len(new_losses)-1, -1, -1):
                if restart_mask[i]:
                    popped += 1
                    print("Removing sigma value:", sigmas[i], "with loss:", new_losses[i])
                    sigmas = jnp.concat([sigmas[:i],sigmas[i+1:]])
                    ks = jnp.concat([ks[:i],ks[i+1:]])
                    neuronss.pop(i)
                    opt_states.pop(i)
                    new_losses.pop(i)
                    old_losses.pop(i)
                    restart_mask.pop(i)
                    accuracies.pop(i)
            if len(new_losses) > 1:
                to_pop = new_losses.index(max(new_losses))
                if accuracies[to_pop] == min(accuracies) or new_losses[to_pop] >= 2*min(new_losses):
                    popped += 1
                    print("Removing sigma value:", sigmas[to_pop], "with loss:", new_losses[to_pop])
                    sigmas = jnp.concat([sigmas[:to_pop],sigmas[to_pop+1:]])
                    ks = jnp.concat([ks[:to_pop],ks[to_pop+1:]])
                    neuronss.pop(to_pop)
                    opt_states.pop(to_pop)
                    new_losses.pop(to_pop)
                    old_losses.pop(to_pop)
                    restart_mask.pop(to_pop)
                    accuracies.pop(to_pop)
            print("Losses:")
            print([round(float(new_loss),5) for new_loss in new_losses])
            print("Accuracies:")
            print([round(acc,5) for acc in accuracies])
            iters = 0
end_time = time.time()
print("Took", end_time-init_time, "seconds to get train.")
print("Learnt:\n", output, "\nwith arch:", arch)
circuit = output_circuit(neuronss[final_index])
[print(circ) for circ in circuit]
print(sigmas[final_index])
