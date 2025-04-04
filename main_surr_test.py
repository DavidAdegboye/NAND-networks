import jax
import jax.numpy as jnp
import optax
import random
import itertools
from typing import List, Tuple, Set, Union, Dict
import time
import yaml
from functools import partial

import sys
import os
if os.name == 'nt':  # Windows
    import msvcrt
else:  # Unix-like systems
    import select

with open("set-up.yaml", "r") as f:
    config = yaml.safe_load(f)

new_batches = 0
def get_optional_input_non_blocking():
    global new_batches, max_fan_in
    if os.name == 'nt':  # Windows
        if msvcrt.kbhit():
            user_input = msvcrt.getch().decode('utf-8').strip()
            if 's' in user_input:
                print("Input received")
                return 1
            if 'd' in user_input:
                print("Input received")
                return 2
            if 'a' in user_input:
                print("Input received")
                max_fan_in += 1
    else:  # Unix-like systems
        input_ready, _, _ = select.select([sys.stdin], [], [], 0)  # Non-blocking select
        if input_ready:
            user_input = sys.stdin.readline().strip()
            if 's' in user_input:
                print("Input received")
                return 1
            if 'd' in user_input:
                print("Input received")
                return 2
            if 'a' in user_input:
                print("Input received")
                max_fan_in += 1
    return 0

# defining some types
Network = List[jnp.ndarray]
NeuronShape = List[int]
LayerShape = List[NeuronShape]
NetworkShape = List[LayerShape]
##jax.config.update("jax_traceback_filtering", "off")

print(jax.devices())

add_or_img = config["add_or_img"]

if add_or_img == 'c':
    import custom_util
    inputs, output, ins, outs, num_ins = custom_util.set_up_custom()
elif add_or_img == 'a':
    import adders_util
    inputs, output, ins, outs, num_ins = adders_util.set_up_adders()
else:
    import image_class_resid
    inputs, x_test, output, y_test, num_ins = image_class_resid.set_up_img()
    inputs = jnp.expand_dims(inputs, axis=1)
    x_test = jnp.expand_dims(x_test, axis=1)
    outs = output.shape[1]

def add_second_layers(input: jnp.ndarray, min_fan: int, max_fan: int) -> jnp.ndarray:
    """
    adds extra bits to the input to help aid in training. These extra bits are produced by putting the current input into
    NAND gates. most commony used wit min_fan=1, max_fan=1, to add a complement layer

    Parameters
    input - an individual input
    min_fan - the minimum fan-in of each NAND gate
    max_fan - the maximum fan-in of each NAND gate
    
    Returns
    output - the input, plus those extra bits
    """
    # giving the network the second layer for free. Can hypothetically do this n times, although its cost grows exponentially.
    output = list(input)
    unchanged = output.copy()
    for k in range(min_fan, max_fan+1):
        for comb in itertools.combinations(unchanged, k):
            output.append(1-jnp.prod(jnp.array(comb)))
    return jnp.array(output)

# adding extra bits to the input to help with learning
if add_or_img == 'i':
    # for images, this is convolutional layers
    convs = config["convs"]
    convs = [[w,s,c+2,ns] for w,s,c,ns in convs]
    true_arch = [config["size"]**2] + [ns**2 for _,_,_,ns in convs]
    inputs = jnp.concatenate([inputs, 1-inputs], axis=1)
    x_test = jnp.concatenate([x_test, 1-x_test], axis=1)
    if convs:
        scaled_train_imgs, scaled_test_imgs = image_class_resid.get_imgs(convs)
        new_ins = convs[-1][2] * convs[-1][3]**2 * 2
    else:
        new_ins = true_arch[0] * 2
        scaled_train_imgs, scaled_test_imgs = [], []
else:
    true_arch = []
    # for adders and arbitrary combinational logic circuits, we're first adding extra layers
    extra_layers = config["extra_layers"]
    for min_fan, max_fan in extra_layers:
        old_ins = inputs.shape[1]
        inputs = jax.vmap(add_second_layers, in_axes=(0, None, None))(inputs, min_fan, max_fan)
        mask = jnp.sum(inputs, axis=0) < 2**ins
        inputs = inputs[:, mask]
        new_ins = inputs.shape[1]
        true_arch.append(new_ins - old_ins)
    # and then if it's an adder, we're also adding extra help for adders
    if add_or_img == 'a':
        inputs, true_arch, add_adder_help, with_nots = adders_util.adder_help(inputs, true_arch)
        use_surr = config["use_surr"]
        if use_surr:
            surr_arr = adders_util.update_surr_arr()
    new_ins = inputs.shape[1]
print(true_arch)

# I found using the same distribution for weights globally, and a sigma value of 1.5 works best, particularly for large networks
# which is what I'm setting up here for images. Still however adding the option for fine tuning for the other ones.
global_weights = config["global_weights"]
sigma_i = config["sigma_i"]

# I've found linear works the best for adders, although there may be a different way to taper down I've not tried.
taper_q = config["taper_q"]
if taper_q == 't':
    taper = config["taper"]
    next_layer = config["width"]
    arch = [new_ins]
    for _ in range(config["hidden"]):
        arch.append(next_layer)
        next_layer = max(min(next_layer-1, round(next_layer*taper)), outs)
    arch.append(outs)
elif taper_q == 'c':
    arch = [new_ins]
    arch += config["architecture"]
    arch += [outs]
elif taper_q == 'f':
    width = config["width"]
    hidden = config["hidden"]
    arch = [new_ins] + [width] * hidden + [outs]
else:
    starting_width = config["width"]
    hidden = config["hidden"]
    diff = starting_width - outs
    layer_diff = diff/hidden
    arch = [new_ins] + [round(starting_width-i*layer_diff) for i in range(hidden)] + [outs]

print("True arch:", true_arch)
print(arch)
true_arch = arch.copy()

if use_surr:
    for i, layer in enumerate(surr_arr):
        if i < len(arch) - 2:
            true_arch[i+1] += len(layer)

print("True arch:", true_arch)

temperature = config["temperature"]
l2_coeff = config["l2_coeff"]
if l2_coeff == 0:
    max_fan_in = 0
else:
    max_fan_in = config["max_fan"]
l3_coeff = config["l3_coeff"]
if l3_coeff == 0:
    max_gates = jnp.array([0]*len(arch))
else:
    max_gates = jnp.array(config["max_gates"])
l3_coeff = l3_coeff / (sum(arch)-sum(max_gates))
l4_coeff = config["l4_coeff"]
l5_coeff = config["l5_coeff"]
if l5_coeff == 0:
    min_gates = jnp.array([0]*len(arch))
else:
    min_gates = jnp.array(config["max_gates"])
    l5_coeff = float(l5_coeff / (sum(min_gates)))
# for adders and arbitrary combinational logic circuits, where we're aiming for 100% accuracy, if we're stuck
# in the high nineties at a local minima, I've added this to give a little nudge. It makes the losses of the
# incorrect samples weigh more.
l6_coeff = config["l6_coeff"]
if l6_coeff == 0:
    mean_fan_in = 0
else:
    mean_fan_in = config["mean_fan"]
weigh_even = 'n'

batches = config["batches"]
batch_size = num_ins//batches



@jax.jit
def f(x: jnp.ndarray, w: jnp.ndarray) -> float:
    """
    Helper function for forward, calculates the continuous effective input a neuron receives from a specific previous layer

    Parameters
    x - could be inputs, could be outputs from a previous NAND gate, importantly it's a jnp array all from the same layer
    w - the weights of those wires connecting x to the NAND gate
    
    Returns
    the continuous effective input from that layer for the NAND gate
    """
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w))

@jax.jit
def f_disc(x: jnp.ndarray, w: jnp.ndarray) -> int:
    """
    Helper function for forward_disc, calculates the discrete effective input a neuron receives from a specific previous layer

    Parameters
    x - could be inputs, could be outputs from a previous NAND gate, importantly it's a 1D jnp array all from the same layer
    w - the weights of those wires connecting x to the NAND gate
    
    Returns
    the discrete effective input from that layer for the NAND gate
    """
    return jnp.prod(jnp.where(w>0, x, 1)) 

@jax.jit
def forward(xs: jnp.ndarray, weights: jnp.ndarray) -> float:
    """
    The continuous forward pass for a neuron

    Parameters
    xs - a 2d jnp array of all the values on those wires
    weights - a 2d jnp array of all the wires going into it
    
    Returns
    the continuous effective output for that NAND gate
    """
    return 1 - jnp.prod(jax.vmap(f)(xs, weights))

@jax.jit
def forward_disc(xs: jnp.ndarray, weights: jnp.ndarray) -> int:
    """
    The discrete forward pass for a neuron

    Parameters
    xs - a 2d jnp array of all the values on those wires
    weights - a 2d jnp array of all the wires going into it
    
    Returns
    the discrete effective output for that NAND gate
    """
    return 1 - jnp.prod(jax.vmap(f_disc)(xs, weights))

def get_used(used: List[int], arch: List[int], verbose: bool) -> List[int]:
    """
    Finds the number of neurons actually used by the network in each layer

    Parameters
    used - a list of the indices of all the neurons used by the network
    arch - the number of neurons in each layer of the network
    
    Returns
    the number of neurons in each layer of the network that are being used
    """
    # finds which how many neurons are actually being used in each layer
    output = []
    current = 0
    layer_i = 0
    current_l = 0
    current_h = arch[0]
    # counting the number of nodes in each layer.
    for node in used:
        if current_l <= node < current_h:
            current += 1
        else:
            while not(current_l <= node < current_h):
                output.append(current)
                current = 0
                layer_i += 1
                current_l = current_h
                current_h = current_l + arch[layer_i]
            current += 1
    output.append(current)
    return output

def clean_connected(connetecteds: Dict[int, List[int]], used_list: List[int], arch: List[int]) -> List[List[jnp.ndarray]]:
    # converts our somewhat clean connecteds dictionary, into a List of jnp arrays representing the learnt NAND network
    upper_bounds = []
    node_count = 0
    for layer in arch:
        node_count += layer
        upper_bounds.append(node_count)
    node_to_true_index = dict()
    layer_index = 0
    current_layer = 0
    layer = []
    net = []
    for node in used_list:
        if node < upper_bounds[current_layer]:
            node_to_true_index[node] = [current_layer, layer_index]
            layer_index += 1
        else:
            while node >= upper_bounds[current_layer]:
                net.append(layer.copy())
                current_layer += 1
                layer_index = 0
                layer = []
            node_to_true_index[node] = [current_layer, layer_index]
            layer_index += 1
        if current_layer != 0:
            connections = [node_to_true_index[con] for con in connetecteds[node]]
            layer.append(connections)
    net.append(layer)
    return net[1:]

def output_circuit(neurons: Network, verbose=True, super_verbose=False) -> List[str]:
    """
    Outputs the learnt circuit, and also prints some useful data about the network
    
    Parameters
    neurons - the internal representation of the circuit as learnt
    verbose - a flag for printing extra info
    
    Returns
    circuits[-arch[-1]:] - a list of the circuit learnt for each output neuron
    """
    # outputs the learnt circuit
    connecteds: List[List[int]] = [[] for _ in range(ins)]
    if extra_layers:
        circuits = [chr(ord('A')+i) for i in range(ins)]
        for layer in extra_layers:
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
        if add_or_img == 'a':
            circuits, connecteds = adders_util.update_circuits(add_adder_help, circuits, with_nots, connecteds)
    else:
        circuits = [chr(ord('A')+i) for i in range(arch[0])]
    gates:List[List[List[Union[str,Tuple[int,int]]]]] = [[[] for _ in range(arch[0])]]
    c2i = dict([(x,i) for i,x in enumerate(circuits)])
    indices = dict([(i,i) for i in range(arch[0])])
    index2gate = dict([(i, (0,i)) for i in range(arch[0])])
    empties = []
    added = arch[0] - 1
    used = set()
    if use_surr:
        sum_arch = []
        sums = 0
        for layer in true_arch:
            sum_arch.append(sums)
            sums += layer
    for layer_i in range(i_1):
        gates.append([])
        gate_i1 = layer_i+1
        gate_i2 = 0
        if use_surr:
            if super_verbose:
                print("surrogates")
            if layer_i < len(surr_arr):
                for neuron_i in range(len(surr_arr[layer_i])):
                    connected: List[Tuple[int, str]] = []
                    for inner_layer_i, weight_i in surr_arr[layer_i][neuron_i]:
                        i = sum_arch[inner_layer_i] + int(weight_i)
                        connected.append((indices[i], circuits[indices[i]]))
                    added += 1
                    connected = sorted(connected)
                    connecteds.append([node[0] for node in connected])
                    i = len(connecteds)-1
                    if super_verbose:
                        print(i, connecteds[i])
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
                            used.add(added)
            if super_verbose:
                print("natties")
        for neuron_i in range(arch[layer_i+1]):
            i = 0
            connected: Set[Tuple[int, str]] = set()
            for inner_layer_i in range(layer_i+1):
                for weight_i in range(true_arch[inner_layer_i]):
                    if neurons[layer_i][neuron_i,inner_layer_i,weight_i] > 0 and indices[i] not in empties:
                        connected.add((indices[i], circuits[indices[i]]))
                    i += 1
            added += 1
            sorted_connected = sorted(list(connected))
            connecteds.append([node[0] for node in sorted_connected])
            i = len(connecteds)-1
            if super_verbose:
                print(i, connecteds[i])
            if not sorted_connected:
                empties.append(added)
                indices[added] = added
                circuits.append('_')
            else:
                if len(sorted_connected) == 1:
                    node = '¬' + sorted_connected[0][1]
                    if len(node) > 2:
                        if node[:3] == "¬¬¬":
                            node = node[2:]
                else:
                    node = '¬(' + '.'.join([element[1] for element in sorted_connected]) + ')'
                if node in c2i.keys():
                    if layer_i == i_1-1:
                        circuits.append(node)
                        gates[-1].append(["=", index2gate[c2i[node]]])
                        index2gate[added] = (gate_i1, gate_i2)
                        gate_i2 += 1
                        for prev_node in sorted_connected:
                            used.add(prev_node[0])
                    else:
                        circuits.append('_')
                    indices[added] = c2i[node]
                else:
                    circuits.append(node)
                    c2i[node] = added
                    indices[added] = added
                    gates[-1].append([index2gate[element[0]] for element in sorted_connected])
                    index2gate[added] = (gate_i1, gate_i2)
                    gate_i2 += 1
                    if layer_i == i_1-1:
                        for prev_node in sorted_connected:
                            used.add(prev_node[0])
                        used.add(added)
    queue = list(used)
    nodes = []
    while len(queue):
        node_i = queue.pop(0)
        nodes.append(node_i)
        for node_2 in connecteds[node_i]:
            if node_2 not in used:
                queue.append(node_2)
                used.add(node_2)
    used_list: List[int] = sorted(list(used))
    if verbose:
        print(used_list)
    true_net = {i: connecteds[i] for i in used_list}
    true_weights = clean_connected(true_net, used_list, true_arch)
    if super_verbose:
        print(true_weights)
    learnt_arch = get_used(used_list, true_arch, verbose)
    fan_ins = []
    for node_index in used_list:
        if node_index >= learnt_arch[0]:
            fan_ins.append(len(connecteds[node_index]))
    with open(f"circuit.txt", "w") as f:
        f.write(f"used:\n{learnt_arch}\nout of:\n{true_arch}\n")
        f.write(f"Max fan-in: {max(fan_ins)}\nAverage fan-in: {round(sum(fan_ins)/len(fan_ins), 2)}\n")
        for circ in circuits[-true_arch[-1]:]:
            f.write(f"{circ}\n")
    print("used:\n", learnt_arch, "\nout of:\n", true_arch)
    print(f"Max fan-in: {max(fan_ins)}\nAverage fan-in: {round(sum(fan_ins)/len(fan_ins), 2)}")
    return circuits[-true_arch[-1]:]

@partial(jax.jit, static_argnames="layer_i")
def calc_surr(xs: jnp.ndarray, layer_i: int) -> jnp.ndarray:
    start = [1-jnp.prod(xs[node[:,0], node[:,1]]) for node in surr_arr[layer_i]]
    return jnp.array(start)

@jax.jit
def feed_forward(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the continous output of the network

    Parameters
    inputs - the input data
    neurons - the network
    
    Returns
    the continuous output
    """
    xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
    for layer_i in range(min(i_1-1, 3)):
        next = jax.vmap(forward, in_axes=(None, 0))(xs, neurons[layer_i])
        if use_surr and layer_i < len(surr_arr):
            next = jnp.concatenate([calc_surr(xs, layer_i), next])
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    for layer_i in range(3, i_1-1):
        next = jax.vmap(forward, in_axes=(None, 0))(xs[jnp.array([0,-2,-1])], neurons[layer_i])
        if use_surr and layer_i < len(surr_arr):
            next = jnp.concatenate([calc_surr(xs, layer_i), next])
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    return jax.vmap(forward, in_axes=(None, 0))(xs, neurons[i_1-1])[:outs]

@jax.jit
def feed_forward_disc(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the discrete output of the network

    Parameters
    inputs - the input data
    neurons - the network
    
    Returns
    the discrete output
    """
    xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
    for layer_i in range(min(i_1-1, 3)):
        next = jax.vmap(forward_disc, in_axes=(None, 0))(xs, neurons[layer_i])
        if use_surr and layer_i < len(surr_arr):
            next = jnp.concatenate([calc_surr(xs, layer_i), next])
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    for layer_i in range(3, i_1-1):
        next = jax.vmap(forward_disc, in_axes=(None, 0))(xs[jnp.array([0,-2,-1])], neurons[layer_i])
        if use_surr and layer_i < len(surr_arr):
            next = jnp.concatenate([calc_surr(xs, layer_i), next])
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    return jax.vmap(forward_disc, in_axes=(None, 0))(xs, neurons[i_1-1])[:outs]

@partial(jax.jit, static_argnames='n')
def forward_conv(xs: jnp.ndarray, weights:jnp.ndarray, s: int, n: int) -> jnp.ndarray:
    """
    Applies a filter of width `w` and stride `s` to the input array `xs`.
    
    Parameters:
    xs - an array of shape (old_channels, old_n, old_n), the input data
    weights - an array of shape (channels, n, n, old_channels, w, w), containing the filter weights
    s - the stride of the filter
    n - the new height and width of the picture
    
    Returns:
    An array of shape (channels, n, n), the result of applying the filter.
    """
    w = weights.shape[2]
    old_channels = xs.shape[0]
    channels = jnp.arange(weights.shape[0])
    return 1-jax.vmap(
        lambda c: jax.vmap(
            lambda i: jax.vmap(
                lambda j: f(jax.lax.dynamic_slice(xs, (0, i*s, j*s), (old_channels, w, w)), weights[c])
            )(jnp.arange(n))
        )(jnp.arange(n))
    )(channels)

@partial(jax.jit, static_argnames='n')
def forward_conv_disc(xs: jnp.ndarray, weights:jnp.ndarray, s: int, n: int) -> jnp.ndarray:
    """
    Applies a filter of width `w` and stride `s` to the input array `xs`.
    
    Parameters:
    xs - an array of shape (old_channels, old_n, old_n), the input data
    weights - an array of shape (channels, n, n, old_channels, w, w), containing the filter weights
    s - the stride of the filter
    n - the new height and width of the picture
    
    Returns:
    An array of shape (channels, n, n), the result of applying the filter.
    """
    w = weights.shape[2]
    old_channels = xs.shape[0]
    channels = jnp.arange(weights.shape[0])
    return 1-jax.vmap(
        lambda c: jax.vmap(
            lambda i: jax.vmap(
                lambda j: f_disc(jax.lax.dynamic_slice(xs, (0, i*s, j*s), (old_channels, w, w)), weights[c])
            )(jnp.arange(n))
        )(jnp.arange(n))
    )(channels)

@jax.jit
def feed_forward_conv(xs: jnp.ndarray, weights:jnp.ndarray, imgs_list: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Applies all of the convolutional layers to the input
    
    Parameters:
    xs - an array of shape (n, n), the input data
    weights - the list of weights
    
    Returns:
    The result of applying the convolutional layers, ready to be passed into
    the dense layers
    """
    for i, (ws, (_,_,s,n)) in enumerate(zip(weights, convs)):
        xs = jnp.concatenate([imgs_list[i], forward_conv(xs, ws, s, n)], axis=0)
    return xs

@jax.jit
def feed_forward_conv_disc(xs: jnp.ndarray, weights:jnp.ndarray, imgs_list: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Applies all of the convolutional layers to the input
    
    Parameters:
    xs - an array of shape (n, n), the input data
    weights - the list of weights
    
    Returns:
    The result of applying the convolutional layers, ready to be passed into
    the dense layers
    """
    for i, (ws, (_,_,s,n)) in enumerate(zip(weights, convs)):
        xs = jnp.concatenate([imgs_list[i], forward_conv(xs, ws, s, n)], axis=0)
    return xs

def get_weights_conv(w: int, c: int, old_c: int, sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the weights for a filter

    Parameters
    w - the width of the filter
    c - the number of channels
    old_c - the number of channels in the previous layer
    sigma - the standard deviation of the normal distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical relation isn't clear, so I pass it in separately)
    
    Returns
    a 2d jnp array of the weights, which represents the wires going into a certain neuron
    """
    global key
    # layer lists, each with arch[i] elements
    # so this is a 2D list of floats
    # or a 1D list of jnp arrays
    if global_weights:
        n = global_conv_n
    else:
        n = old_c*w**2
    mu = -jnp.log(n-1)/k
    return sigma * jax.random.normal(jax.random.key(key), shape=(c, old_c, w, w)) + mu #type: ignore

def initialise_conv(convs: List[Tuple[int, int, int, int]], sigma: jnp.ndarray, k: jnp.ndarray) -> Network:
    """
    initialises the network

    Parameters
    convs - a tuple describing the convolutional layers we're adding
    sigma - the standard deviation of the normal distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical relation isn't clear, so I pass it in separately)
    
    Returns
    the convolutional layers of the network
    """
    neurons = []
    current_c = 2
    for w,_,c,_ in convs:
        weights = get_weights_conv(w, c-2, current_c, sigma, k)
        neurons.append(weights)
        current_c = c
    return neurons

def get_weights(layer: int, arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the weights of a given neuron

    Parameters
    layer - the layer that the neuron is in
    arch - a list representing the architecture
    sigma - the standard deviation of the normal distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical relation isn't clear, so I pass it in separately)
    
    Returns
    a 2d jnp array of the weights, which represents the wires going into a certain neuron
    """
    global key
    if layer == 1 or layer == 2 or layer == len(arch)-1:
        weights = jnp.ones((layer, i_4)) * -jnp.inf
    else:
        weights = jnp.ones((3,i_4)) * -jnp.inf
    # layer lists, each with arch[i] elements
    # so this is a 2D list of floats
    # or a 1D list of jnp arrays
    if global_weights:
        n = global_n
    else:
        if layer == 1 or layer == len(arch)-1:
            n = sum(arch[:layer])
        else:
            n = arch[0] + arch[layer-2] + arch[layer-1]
    mu = -jnp.log(n-1)/k
    if layer == 1 or layer == 2 or layer == len(arch)-1:
        for i in range(layer):
            inner_layer = sigma * jax.random.normal(jax.random.key(key), (arch[i])) + mu
            weights = weights.at[i].set(jnp.pad(inner_layer, (0, i_4-arch[i]), mode="constant", constant_values=-jnp.inf))
            key = random.randint(0, 10000)
    else:
        inner_layer = sigma * jax.random.normal(jax.random.key(key), (arch[0])) + mu
        weights = weights.at[0].set(jnp.pad(inner_layer, (0, i_4-arch[0]), mode="constant", constant_values=-jnp.inf))
        key = random.randint(0, 10000)
        for i in range(1,3):
            inner_layer = sigma * jax.random.normal(jax.random.key(key), (arch[layer-3+i])) + mu
            weights = weights.at[i].set(jnp.pad(inner_layer, (0, i_4-arch[layer-3+i]), mode="constant", constant_values=-jnp.inf))
            key = random.randint(0, 10000)
    return weights

def initialise(arch: List[int], true_arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> List[jnp.ndarray]:
    """
    initialises the network

    Parameters
    arch - a list representing the architecture
    sigma - the standard deviation of the normal distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical relation isn't clear, so I pass it in separately)
    
    Returns
    the network
    """
    neurons = []
    for i1 in range(1, len(arch)):
        if i1 == 1 or i1 == 2 or i1 == len(arch) - 1:
            layer = jnp.ones((arch[i1], i1, i_4))
        else:
            layer = jnp.ones((arch[i1], 3, i_4))
        for i2 in range(arch[i1]):
            layer = layer.at[i2].set(get_weights(i1, true_arch, sigma, k))
        neurons.append(layer)
    return neurons

@jax.jit
def get_l2(neurons: Network, max_fan_in: int) -> float:
    """
    calculates l2, which is minimised for any maximum fan-in under or equal to "max_fan_in"
    this doesn't account for duplicate gates

    Parameters
    neurons - the network
    
    Returns
    l2
    """
    fan_ins = jnp.array([])
    for layer in neurons:
        fan_ins = jnp.concatenate((fan_ins, jax.vmap(lambda x:jnp.sum(jax.nn.sigmoid(x/temperature)))(layer)))
    l2s = jax.nn.relu(fan_ins-max_fan_in)
    return jnp.sum(jax.nn.softmax(l2s)*l2s)

num_neurons = sum(true_arch[1:])
@jax.jit
def get_l6(neurons: Network, mean_fan_in: float) -> float:
    """
    calculates l2, which is minimised for any maximum fan-in under or equal to "max_fan_in"
    this doesn't account for duplicate gates

    Parameters
    neurons - the network
    
    Returns
    l2
    """
    fan_ins = jnp.array([])
    for layer in neurons:
        fan_ins = jnp.concatenate((fan_ins, jax.vmap(lambda x:jnp.sum(jax.nn.sigmoid(x/temperature)))(layer)))
    l6s = jnp.sum(fan_ins)/num_neurons
    return jax.nn.relu(l6s-mean_fan_in)

@jax.jit
def get_l2_disc(neurons: Network, max_fan_in: int) -> float:
    """
    calculates l2, which is minimised for any maximum fan-in under or equal to "max_fan_in"
    this doesn't account for duplicate gates

    Parameters
    neurons - the network
    
    Returns
    l2
    """
    fan_ins = jnp.array([])
    for layer in neurons:
        fan_ins = jnp.concatenate((fan_ins, jax.vmap(lambda x:jnp.sum(jnp.where(x>0, 1, 0)))(layer)))
    l2s = jax.nn.relu(fan_ins-max_fan_in)
    return jnp.max(l2s)

@partial(jax.jit, static_argnums=1)
def cont_or_arr(arr: jnp.ndarray, axis=None) -> jnp.ndarray:
    # computes a continuous or, using De Morgan's
    return 1-jnp.prod(1-arr, axis=axis)

@jax.jit
def cont_or(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    return 1-(1-arr1)*(1-arr2)

def func(layer: int) -> jnp.ndarray:
    if layer <= 2 or layer == len(arch) - 1:
        return jnp.arange(layer)
    return jnp.array([0, layer-2, layer-1])

@jax.jit
def get_l3_used(neurons: Network) -> float:
    """
    calculates l3, which is minimised for any number of gates less than or equal to "max_gates"
    used_back tells us if the NAND gates will be used in the outputs
    used_for tells us if the NAND gates are connected to the inputs
    and so their product tells us if the NAND gates are being used
    just like l2, this doesn't account for duplicate gates (but output_circuit does)

    Parameters
    neurons - the network
    
    Returns
    l3
    """
    sig_neurons = [jax.nn.sigmoid(layer/temperature) for layer in neurons]
    used_back = jnp.zeros(shape=(len(arch), i_4))
    used_back = used_back.at[len(arch)-1, :outs].set(jnp.ones(shape=outs))
    # outputs are used by outputs
    for layer in range(len(arch)-1, 0, -1):
        temp = sig_neurons[layer-1] * used_back[layer, :arch[layer]][:, jnp.newaxis, jnp.newaxis]
        # this is a 2D matrix, the LHS of the * is how much each neuron to the left of this neuron is used by this neuron
        # the RHS of the * is a vector, which is how much this neuron is used by the output.
        temp = cont_or_arr(temp, axis=0)
        used_back = used_back.at[func(layer)].set(cont_or(used_back[func(layer)], temp))
    used_for = jnp.zeros(shape=(len(arch), i_4))
    used_for = used_for.at[0, :new_ins].set(jnp.ones(shape=new_ins))
    for layer in range(1, len(arch)):
        temp = sig_neurons[layer-1][:arch[layer]] * used_for[func(layer)][jnp.newaxis,:,:]
        temp = cont_or_arr(temp, axis=(1,2))
        used_for = used_for.at[layer, :arch[layer]].set(cont_or(used_for[layer, :arch[layer]], temp))
    return used_back*used_for

@jax.jit
def get_l3(neurons: Network, max_gates: jnp.ndarray) -> float:
    used = get_l3_used(neurons)
    return jnp.sum(jax.nn.relu(jnp.sum(used, axis=1)-max_gates))

@jax.jit
def print_l3(neurons: Network) -> float:
    used = get_l3_used(neurons)
    return jnp.sum(used, axis=1)

# @jax.jit
def print_l3_disc(neurons: Network) -> float:
    """
    calculates l3, which is minimised for any number of gates less than or equal to "max_gates"
    used_back tells us if the NAND gates will be used in the outputs
    used_for tells us if the NAND gates are connected to the inputs
    and so their product tells us if the NAND gates are being used
    just like l2, this doesn't account for duplicate gates (but output_circuit does)

    Parameters
    neurons - the network
    
    Returns
    l3
    """
    sig_neurons = [jnp.where(layer>0,1,0) for layer in neurons]
    used_back = jnp.zeros(shape=(len(arch), i_4))
    used_back = used_back.at[len(arch)-1, :outs].set(jnp.ones(shape=outs))
    # outputs are used by outputs
    for layer in range(len(arch)-1, 0, -1):
        temp = sig_neurons[layer-1] * used_back[layer, :arch[layer]][:, jnp.newaxis, jnp.newaxis]
        # this is a 2D matrix, the LHS of the * is how much each neuron to the left of this neuron is used by this neuron
        # the RHS of the * is a vector, which is how much this neuron is used by the output.
        temp = cont_or_arr(temp, axis=0)
        used_back = used_back.at[func(layer)].set(cont_or(used_back[func(layer)], temp))
    used_for = jnp.zeros(shape=(len(arch), i_4))
    used_for = used_for.at[0, :new_ins].set(jnp.ones(shape=new_ins))
    for layer in range(1, len(arch)):
        temp = sig_neurons[layer-1][:arch[layer]] * used_for[func(layer)][jnp.newaxis,:,:]
        temp = cont_or_arr(temp, axis=(1,2))
        used_for = used_for.at[layer, :arch[layer]].set(cont_or(used_for[layer, :arch[layer]], temp))
    return jnp.sum(used_back*used_for, axis=1)

@jax.jit
def get_l4(neurons: Network) -> float:
    """
    calculates l4, which is minimised when the weights are close to +- infinity
    essentially a factor that nudges the network to be certain, or to discretize the network
    this should help in ensuring that a low loss corresponds to a high accuracy
    by minimising the difference between the continous and discrete network
    """
    s = 0
    for layer in neurons:
        s += jnp.sum(1-jax.nn.sigmoid(jnp.absolute(layer)))
    return s

@jax.jit
def get_l5(neurons: Network, min_gates: jnp.ndarray) -> float:
    used = get_l3_used(neurons)
    return jnp.sum(jax.nn.relu(min_gates-jnp.sum(used, axis=1)))

epsilon = 1e-7
@partial(jax.jit, static_argnames="l5_coeff")
def loss(neurons: Network, inputs: jnp.ndarray, output: jnp.ndarray, mask1: jnp.ndarray, mask2:jnp.ndarray, max_fan_in: int, max_gates: jnp.ndarray, l5_coeff: float) -> float:
    """
    calculates loss

    Parameters
    neurons - the network
    inputs - all of the inputs (training xs)
    output - all of the outputs (training labels or ys)
    mask1 - a mask for samples we got right
    mask2 - a mask for the samples we got wrong
    
    Returns
    loss
    """
    pred = jax.vmap(feed_forward, in_axes=(0, None))(inputs, neurons)
    pred = jnp.clip(pred, epsilon, 1-epsilon)
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    if weigh_even == 'y':
        # separately calculating the loss of the correct and incorrect outputs. Weighs the rarer one (which should be incorrect)
        # more heavily. The idea is to give it a bigger nudge to overcome a small final hurdle.
        l11 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits[mask1], output[mask1]))
        l12 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits[mask2], output[mask2]))
        s1 = jnp.size(mask1)
        s2 = jnp.size(mask2)
        x = s1/(s1+s2)
        # scaling_factor = 4 * x * (1 - x)
        scaling_factor = 2 * (0.25 - (x-0.5)**2) ** 0.5
        # x=0.5 leads to a scaling factor of 1
        # with minimums at 0,0 and 1,0.
        c1 = x + (0.5 - x) * scaling_factor
        # pulling x towards 0.5 by the scaling factor.
        c2 = 1-c1
        l1 = (c1*l11 + c2*l12)/2
    else:
        l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    if l2_coeff:
        l1 += l2_coeff * get_l2(neurons, max_fan_in)
    if l3_coeff:
        l1 += l3_coeff * get_l3(neurons, max_gates)
    if l4_coeff:
        l1 += l4_coeff * get_l4(neurons)
    if l5_coeff:
        l1 += l5_coeff * get_l5(neurons, min_gates)
    if l6_coeff:
        l1 += l6_coeff * get_l6(neurons, mean_fan_in)
    return l1

@partial(jax.jit, static_argnames="l5_coeff")
def loss_conv(network: List[Network], inputs: jnp.ndarray, output: jnp.ndarray, max_fan_in: int, l5_coeff: float, scaled: List[jnp.ndarray]) -> float:
    """
    calculates loss

    Parameters
    network - [neurons, neurons_conv], where neurons are the dense layers, and neurons_conv are the convolutional
    inputs - all of the inputs (training xs)
    output - all of the outputs (training labels or ys)
    
    Returns
    loss
    """
    if convs:
        pred = jax.vmap(feed_forward_conv, in_axes=(0, None, 0))(inputs, network[1], scaled)
    else:
        inputs = inputs.reshape(inputs.shape[0], -1)
        return loss(network[0], inputs, output, jnp.array([]), jnp.array([]), max_fan_in, max_gates, l5_coeff)
    pred = pred.reshape(pred.shape[0], -1)
    if convs:
        pred = jnp.concatenate([pred, 1-pred], axis=1)
    return loss(network[0], pred, output, jnp.array([]), jnp.array([]), max_fan_in, max_gates, l5_coeff)

@jax.jit
def test(neurons: Network) -> bool:
    """
    is true iff the network is 100% accurate

    Parameters
    neurons - the network
    
    Returns
    if the network was 100% accurate
    """
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neurons)
    return jnp.all(pred==output)

current_max_fan_in = -1
def test_fan_in(neurons: Network) -> bool:
    global current_max_fan_in
    """
    is true iff the max fan-in is less than what the user specified

    Parameters
    neurons - the network
    
    Returns
    if the max fan-in is less than what the user specified
    """
    temp = 0
    for layer in neurons:
        # this can include gates that aren't used and have a fan-in greater
        # so if the circuit printed is better, we can stop the search anyway
        fan_ins = jax.vmap(lambda x:jnp.sum(jnp.where(x>0, 1, 0)))(layer)
        temp = max(temp, jnp.max(fan_ins))
    if temp > max_fan_in:
        if (temp < current_max_fan_in or current_max_fan_in == -1):
            print(temp, max_fan_in)
            [print(circ) for circ in (output_circuit(neurons, True, True))]
            print("Max fan-in not good enough")
            current_max_fan_in = temp
        return False
    return True

# @jax.jit
def acc(neurons: Network) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    calculates the accuracy, and also the masks used in the loss function

    Parameters
    neurons - the network
    
    Returns
    accuracy - the accuracy (may be specifically the testing accuracy)
    mask1 - the mask of the samples it got right
    mask2 - the mask of the samples it got wrong
    """
    # returns the accuracy
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neurons)
    pred = (pred == output)
    pred = jnp.sum(pred, axis=1)
    if weigh_even == 'y':
        trues = jnp.where(pred == outs)
        falses = jnp.where(pred < outs)
        return jnp.sum(pred)/((2**(ins))*(outs)), trues[0], falses[0]
    return jnp.sum(pred)/((2**(ins))*(outs)), jnp.zeros(0), jnp.zeros(0)

@jax.jit
def acc_conv(neurons: Network, neurons_conv: Network) -> List[float]:
    """
    calculates the accuracy, and also the masks used in the loss function

    Parameters
    neurons - the dense layers
    neurons_conv - the convolutional layers
    
    Returns
    accuracy - the accuracy (may be specifically the testing accuracy)
    """
    # returns the accuracy
    if convs:
        pred = jax.vmap(feed_forward_conv_disc, in_axes=(0, None, 0))(x_test, neurons_conv, scaled_test_imgs)
        pred = jnp.concatenate([pred, 1-pred], axis=1)
    else:
        pred = x_test
    pred = pred.reshape(pred.shape[0], -1)
    # print(jnp.sum(pred))
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(pred, neurons)
    result = jax.vmap(image_class_resid.evaluate)(pred, y_test)
    return jnp.sum(result)/result.size

all_sigmas = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
all_ks = [1.0, 0.99, 0.98, 0.97, 0.955, 0.94, 0.92, 0.91, 0.9, 0.85, 0.75, 0.65, 0.5, 0.39, 0.32, 0.27, 0.23, 0.205, 0.18, 0.17, 0.155, 0.14, 0.13, 0.12, 0.11]

def start_run(batches, batch_size):
    global i_1, i_3, i_4, global_n, global_conv_n, boundary_jump, schedule, solver, key, neurons, neurons_conv, opt_state, grad, grad_conv
    i_1 = len(true_arch) - 1
    # i_2 = max(true_arch[1:])
    i_3 = i_1
    i_4 = max(true_arch)
    neurons_shape = []
    for i in range(1, len(true_arch)):
        if i <= 3 or i == len(true_arch)-1:
            neurons_shape.append((sum(true_arch[:i]), true_arch[i]))
        else:
            neurons_shape.append((true_arch[0]+true_arch[i-2]+true_arch[i-1], true_arch[i]))
    global_n = sum(ns[0]*ns[1] for ns in neurons_shape)/sum(ns[1] for ns in neurons_shape)
    if add_or_img == 'i' and convs:
        neurons_conv_shape = []
        # a list for the convolutional layers
        # of the number of inputs per NAND gate, and the number of NAND gates in that layer
        old_c = 2
        for w,_,c,ns in convs:
            neurons_conv_shape.append((w**2*old_c, c*ns**2))
            old_c = c
        neurons_shape = []
        for i in range(1, len(arch)):
            if i <= 3 or i == len(arch)-1:
                neurons_shape.append((sum(arch[:i]), arch[i]))
            else:
                neurons_shape.append((arch[0]+arch[i-2]+arch[i-1], arch[i]))
        global_n = (sum([ncs[0]*ncs[1] for ncs in neurons_conv_shape])+
                    sum(ns[0]*ns[1] for ns in neurons_shape))/(
                        sum([ncs[1] for ncs in neurons_conv_shape])
                        +sum(ns[1] for ns in neurons_shape))
        global_conv_n = global_n
        global_conv_n = sum([ncs[0]*ncs[1] for ncs in neurons_conv_shape])/sum([ncs[1] for ncs in neurons_conv_shape])
        global_n = sum(ns[0]*ns[1] for ns in neurons_shape)/sum(ns[1] for ns in neurons_shape)

    boundary_jump = 5*(max(10//batches,1)**2)*batch_size
    lr_multiplier = batch_size**0.5

    schedule = optax.join_schedules(
        schedules=[
            # optax.constant_schedule(1.0),
            # optax.constant_schedule(0.1*lr_multiplier),
            # optax.constant_schedule(0.03*lr_multiplier),
            # optax.constant_schedule(0.01*lr_multiplier),
            optax.constant_schedule(0.003*lr_multiplier),
            # optax.constant_schedule(0.001*lr_multiplier),
        ],
        boundaries=[(i+1)**2*boundary_jump for i in range(2)]
    )

    solver = optax.adam(learning_rate=schedule)

    print("Learning:\n", output, "\nwith arch:", true_arch)
    key = random.randint(0, 10000)
    start_time = time.time()
    if add_or_img == 'i':
        neurons = initialise(arch, true_arch, all_sigmas[sigma_i], all_ks[sigma_i])
        neurons_conv = initialise_conv(convs, all_sigmas[4], all_ks[4])
    else:
        neurons = initialise(arch, true_arch, all_sigmas[sigma_i], all_ks[sigma_i])
    if add_or_img == 'i':
        opt_state = solver.init([neurons, neurons_conv])
    else:
        opt_state = solver.init(neurons)
    init_time = time.time()
    print("Took", init_time-start_time, "seconds to initialise.")
    print([layer.shape for layer in neurons])
    if add_or_img == 'i' and convs:
        print([layer.shape for layer in neurons_conv])
    if add_or_img == 'i':
        accuracy = acc_conv(neurons, neurons_conv)
        new_loss = loss_conv([neurons, neurons_conv], inputs, output, max_fan_in, l5_coeff, scaled_train_imgs)
        print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),5)}")
        print(print_l3(neurons))
        print(print_l3_disc(neurons))
        print(get_l2(neurons, max_fan_in), get_l2_disc(neurons, max_fan_in), max_fan_in)
        grad_conv = jax.jit(jax.grad(loss_conv, argnums=0), static_argnames=["l5_coeff"])
    else:
        accuracy = acc(neurons)
        new_loss = loss(neurons, inputs, output, jnp.array([]), jnp.array([]), max_fan_in, max_gates, l5_coeff)
        print(f"Accuracy: {round(100*float(accuracy[0]),2)}%, Loss: {round(float(new_loss),5)}")
        print(print_l3(neurons))
        print(print_l3_disc(neurons))
        print(get_l2(neurons, max_fan_in), get_l2_disc(neurons, max_fan_in), max_fan_in)
        print(get_l3(neurons, max_gates))
        grad = jax.jit(jax.grad(loss, argnums=0), static_argnames=["l5_coeff"])

def run(timeout=config["timeout"]):
    global batches, batch_size, inputs, output, weigh_even, neurons, neurons_conv, updates, opt_state, l5_coeff, scaled_train_imgs
    cont = True
    iters = 0
    file_i = -1
    start_run_time = time.time()
    while cont:
        iters += 1
        for _ in range(max(10//batches, 1)):
            if batches > 1:
                key = random.randint(0, 10000)
                key = jax.random.PRNGKey(key)
                shuffled_indices = jax.random.permutation(key, inputs.shape[0])
                inputs = inputs[shuffled_indices]
                output = output[shuffled_indices]
                if add_or_img == 'i' and convs:
                    scaled_train_imgs = [imgs[shuffled_indices] for imgs in scaled_train_imgs]
            # batched_inputs = inputs.reshape(batches, batch_size, inputs.shape[1])
            # batched_output = output.reshape(batches, batch_size, output.shape[1])
            for batch in range(batches):
                # print(batch)
                if add_or_img == 'i':
                    gradients = grad_conv([neurons, neurons_conv],
                                        inputs[batch*batch_size:(batch+1)*batch_size],
                                        output[batch*batch_size:(batch+1)*batch_size],
                                        max_fan_in, l5_coeff,
                                        [imgs[batch*batch_size:(batch+1)*batch_size] for imgs in scaled_train_imgs])
                    update, opt_state = solver.update(gradients, opt_state, [neurons, neurons_conv])
                    neurons, neurons_conv = optax.apply_updates([neurons, neurons_conv], update)
                elif weigh_even == 'y':
                    gradients = grad(neurons,
                                    inputs[batch*batch_size:(batch+1)*batch_size],
                                    output[batch*batch_size:(batch+1)*batch_size],
                                    accuracy[1][batch*batch_size:(batch+1)*batch_size],
                                    accuracy[2][batch*batch_size:(batch+1)*batch_size],
                                    max_fan_in, max_gates, l5_coeff)
                    updates, opt_state = solver.update(gradients, opt_state, neurons)
                    neurons = optax.apply_updates(neurons, updates)
                    accuracy = acc(neurons)
                else:
                    gradients = grad(neurons,
                                    inputs[batch*batch_size:(batch+1)*batch_size],
                                    output[batch*batch_size:(batch+1)*batch_size],
                                    jnp.array([]), jnp.array([]), max_fan_in, max_gates, l5_coeff)
                    updates, opt_state = solver.update(gradients, opt_state, neurons)
                    neurons = optax.apply_updates(neurons, updates)
            if time.time() - start_run_time > timeout * 60:
                if add_or_img == 'i':
                    new_loss = loss_conv([neurons, neurons_conv], inputs, output, max_fan_in, l5_coeff, scaled_train_imgs)
                else:
                    if weigh_even == 'y':
                        new_loss = loss(neurons, inputs, output, accuracy[1], accuracy[2], max_fan_in, max_gates, l5_coeff)
                    else:
                        new_loss = loss(neurons, inputs, output, jnp.array([]), jnp.array([]), max_fan_in, max_gates, l5_coeff)
                if add_or_img == 'i':
                    accuracy = acc_conv(neurons, neurons_conv)
                    print(f"Accuracy: {str(round(100*float(accuracy),2))}%, Loss: {round(float(new_loss),5)}")
                    print(print_l3(neurons))
                    print(print_l3_disc(neurons))
                    print(get_l2(neurons, max_fan_in), get_l2_disc(neurons, max_fan_in), max_fan_in)
                    image_class_resid.save(neurons, convs, str(round(float(100*accuracy[0]),2))+'%', file_i)
                    return accuracy
                else:
                    accuracy = acc(neurons)
                    print(f"Accuracy: {round(100*float(accuracy[0]),2)}%, Loss: {round(float(new_loss),5)}")
                    print(print_l3(neurons))
                    print(print_l3_disc(neurons))
                    print(get_l2(neurons, max_fan_in), get_l2_disc(neurons, max_fan_in), max_fan_in)
                    print(get_l3(neurons, max_gates))
                    circuit = output_circuit(neurons, verbose=True, super_verbose=True)
                    [print(circ) for circ in circuit]
                    return accuracy[0]
        if get_optional_input_non_blocking() == 4:
            batches = new_batches
            batch_size = num_ins // batches
        if add_or_img != 'i':
            if test(neurons) and (l2_coeff==0 or test_fan_in(neurons)) or get_optional_input_non_blocking() == 2:
                cont = False
        if cont:
            if config["l5_coeff"] > 0 and l5_coeff != 0:
                if get_l5(neurons, min_gates, l5_coeff) < 1:
                    print("Setting l5_coeff to 0")
                    l5_coeff = 0
            if add_or_img == 'i':
                new_loss = loss_conv([neurons, neurons_conv], inputs, output, max_fan_in, l5_coeff, scaled_train_imgs)
            else:
                if weigh_even == 'y':
                    new_loss = loss(neurons, inputs, output, accuracy[1], accuracy[2], max_fan_in, max_gates, l5_coeff)
                else:
                    new_loss = loss(neurons, inputs, output, jnp.array([]), jnp.array([]), max_fan_in, max_gates, l5_coeff)
                if get_optional_input_non_blocking() == 1:
                    if add_or_img == 'i':
                        cont = False
                        print("Done training!")
                        print("Testing on testing data...")
                        accuracy = acc_conv(neurons, neurons_conv)
                        print(f"Accuracy: {str(round(100*float(accuracy),2))}%, Loss: {round(float(new_loss),5)}")
                        print(print_l3(neurons))
                        print(print_l3_disc(neurons))
                        print(get_l2(neurons, max_fan_in), get_l2_disc(neurons, max_fan_in), max_fan_in)
                        image_class_resid.save(neurons, convs, str(round(float(100*accuracy[0]),2))+'%', file_i)
                    elif weigh_even == 'n':
                        print("Now weighing wrong more")
                        weigh_even = 'y'
                    else:
                        print("Now weighing wrong less again")
                        weigh_even = 'n'
                elif get_optional_input_non_blocking() == 3:
                    if weigh_even == 'y':
                        weigh_even = 'n'
            if iters == max(10//batches, 1):
                if add_or_img != 'i':
                    accuracy = acc(neurons)
                    print(f"Accuracy: {round(100*float(accuracy[0]),2)}%, Loss: {round(float(new_loss),5)}")
                    print(print_l3(neurons))
                    print(print_l3_disc(neurons))
                    print(get_l2(neurons, max_fan_in), get_l2_disc(neurons, max_fan_in), max_fan_in)
                    print(get_l3(neurons, max_gates))
                else:
                    accuracy = acc_conv(neurons, neurons_conv)
                    print(f"Accuracy: {str(round(100*float(accuracy),2))}%, Loss: {round(float(new_loss),5)}")
                    print(print_l3(neurons))
                    print(print_l3_disc(neurons))
                    print(get_l2(neurons, max_fan_in), get_l2_disc(neurons, max_fan_in), max_fan_in)
                    file_i = image_class_resid.save(arch, neurons_conv, neurons, convs, str(round(float(100*accuracy),2))+'%', file_i)
                iters = 0
    end_time = time.time()
    print("Took", end_time-start_run_time, "seconds to train.")
    if add_or_img != 'i':
        circuit = output_circuit(neurons, True, True)
        [print(circ) for circ in circuit]
    return accuracy[0]

start_run(batches, batch_size)
run()

batch_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
l2_coeffs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
max_fan_ins = [8, 16]

batch_list = [8192, 1024, 128, 65536]
l2_coeffs = [0.1, 0.5, 1.0]
temperatures = [0.2, 0.5, 1.0]
max_fan_ins = [8, 16]
time_to_run = 10

# with open(f"results.txt", "w") as results:
#     results.write(f"ran each for {time_to_run} minutes.\n")

# for b in batch_list:
#     batches = b
#     batch_size = num_ins // b
#     for l in l2_coeffs:
#         l2_coeff = l
#         for mf in max_fan_ins:
#             max_fan_in = mf
#             if l:
#                 for t in temperatures:
#                     temperature = t
#                     print(f"{batches} batches, l2_coeff={l2_coeff}, max fan-in={max_fan_in}, temperature={temperature}")
#                     start_run(arch, batches, batch_size)
#                     accuracy = run(time_to_run)
#                     with open(f"results.txt", "a") as results:
#                         results.write(f"{batches} batches, l2_coeff={l2_coeff}, max fan-in={max_fan_in}, temperature={temperature}, acc:{round(100*float(accuracy),2)}%\n")
#             else:
#                 temperature = 0.2
#                 print(f"{batches} batches, l2_coeff={l2_coeff}, max fan-in={max_fan_in}, temperature={temperature}")
#                 start_run(arch, batches, batch_size)
#                 accuracy = run(time_to_run)
#                 with open(f"results.txt", "a") as results:
#                     results.write(f"{batches} batches, l2_coeff={l2_coeff}, max fan-in={max_fan_in}, temperature={temperature}, acc:{round(100*float(accuracy),2)}%\n")
