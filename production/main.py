import jax
import jax.numpy as jnp
import optax
import random
import itertools
from typing import List, Tuple, Set, Union, Dict, Callable
import time
import yaml
import jax.scipy.special as jsp_special
from functools import partial

import sys
import os
if os.name == 'nt':  # Windows
    import msvcrt
else:  # Unix-like systems
    import select

with open("set-up.yaml", "r") as f:
    config = yaml.safe_load(f)

jax.config.update("jax_traceback_filtering", config["traceback"])

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
Network = Tuple[jnp.ndarray, ...]

print(jax.devices())

add_img_or_custom = config["add_img_or_custom"]

if add_img_or_custom == 'c':
    import utils.custom_util as custom_util
    inputs, output, ins, outs, num_ins = custom_util.set_up_custom()
elif add_img_or_custom == 'a':
    import utils.adders_util as adders_util
    inputs, output, ins, outs, num_ins = adders_util.set_up_adders()
else:
    import utils.image_util as image_util
    inputs, x_test, output, y_test, num_ins = image_util.set_up_img()
    inputs = jnp.expand_dims(inputs, axis=1)
    x_test = jnp.expand_dims(x_test, axis=1)
    outs = output.shape[1]

ALL_SIGMAS = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
              0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
              8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
ALL_KS = [1.0, 0.99, 0.98, 0.97, 0.955, 0.94, 0.92, 0.91,
          0.9, 0.85, 0.75, 0.65, 0.5, 0.39, 0.32, 0.27, 0.23,
          0.205, 0.18, 0.17, 0.155, 0.14, 0.13, 0.12, 0.11]

def add_second_layers(input: jnp.ndarray, min_fan: int, max_fan: int
                      ) -> jnp.ndarray:
    """
    adds extra bits to the input to help aid in training. These extra bits are
    produced by putting the current input into NAND gates. most commony used 
    with min_fan=1, max_fan=1, to add a complement layer

    Parameters
    input - an individual input
    min_fan - the minimum fan-in of each NAND gate
    max_fan - the maximum fan-in of each NAND gate
    
    Returns
    output - the input, plus those extra bits
    """
    output = list(input)
    unchanged = output.copy()
    for k in range(min_fan, max_fan+1):
        for comb in itertools.combinations(unchanged, k):
            output.append(1-jnp.prod(jnp.array(comb)))
    return jnp.array(output)

# adding extra bits to the input to help with learning
if add_img_or_custom == 'i':
    # for images, this is convolutional layers
    convs = config["convs"]
    convs = [[w,s,2*c+2,ns] for w,s,c,ns in convs]
    true_arch = [config["size"]**2] + [ns**2 for _,_,_,ns in convs]
    inputs = jnp.concatenate([inputs, 1-inputs], axis=1)
    x_test = jnp.concatenate([x_test, 1-x_test], axis=1)
    if convs:
        scaled_train_imgs, scaled_test_imgs = image_util.get_imgs(convs)
        new_ins = convs[-1][2] * convs[-1][3]**2
    else:
        pools, pool_tests = image_util.get_pools(config["pool_filters"])
        inputs = jnp.concatenate(
            [arr.reshape(arr.shape[0], -1) for arr in pools], axis=1)
        x_test = jnp.concatenate(
            [arr.reshape(arr.shape[0], -1) for arr in pool_tests], axis=1)
        true_arch = [config["size"]**2,inputs.shape[1]]
        new_ins = true_arch[1]
        scaled_train_imgs, scaled_test_imgs = (), ()
    use_surr = False
    surr_arr = None
else:
    true_arch = []
    # for adders and arbitrary combinational logic circuits
    # we're first adding extra layers
    extra_layers = config["extra_layers"]
    for min_fan, max_fan in extra_layers:
        old_ins = inputs.shape[1]
        inputs = jax.vmap(add_second_layers, in_axes=(0, None, None))(
            inputs, min_fan, max_fan)
        mask = jnp.sum(inputs, axis=0) < 2**ins
        inputs = inputs[:, mask]
        new_ins = inputs.shape[1]
        true_arch.append(new_ins - old_ins)
    # and then if it's an adder, we're also adding extra help for adders
    if add_img_or_custom == 'a':
        inputs, true_arch, add_adder_help, with_nots = adders_util.adder_help(
            inputs, true_arch)
        use_surr = config["use_surr"]
        if use_surr:
            surr_arr = adders_util.update_surr_arr()
        else:
            surr_arr = None
    else:
        use_surr = False
        surr_arr = None
    new_ins = inputs.shape[1]
print(true_arch)

global_weights = config["global_weights"]
dense_sigma = config["dense_sigma"]
dense_k = config["dense_k"]
if add_img_or_custom == 'i':
    conv_sigma = config["conv_sigma"]
    conv_k = config["conv_k"]

# I've found linear works the best for adders, although there may be a
# different way to taper down I've not tried.
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
    arch = [new_ins] + [
        round(starting_width-i*layer_diff) for i in range(hidden)] + [outs]

true_arch = arch.copy()

if use_surr:
    surr_copy = []
    for i, layer in enumerate(surr_arr):
        if i < len(arch) - 2:
            true_arch[i+1] += len(layer)
            surr_copy.append(layer)
    surr_arr = surr_copy.copy()

i_1 = len(true_arch) - 1
# i_2 = max(true_arch[1:])
# i_3 = i_1
i_4 = max(true_arch)

neurons_shape = []
for i in range(1, len(true_arch)):
    if i <= 3 or i == len(true_arch)-1:
        neurons_shape.append((sum(true_arch[:i]), true_arch[i]))
    else:
        neurons_shape.append(
            (true_arch[0]+true_arch[i-2]+true_arch[i-1], true_arch[i]))
global_n = (sum(ns[0]*ns[1] for ns in neurons_shape) 
            / sum(ns[1] for ns in neurons_shape))

num_neurons = sum(true_arch[1:])
num_wires = sum(ns[0]*ns[1] for ns in neurons_shape)

temperature = config["temperature"]
max_fan_in_penalty_coeff = config["max_fan_in_penalty_coeff"]
if max_fan_in_penalty_coeff == 0:
    max_fan_in = 0
else:
    max_fan_in = config["max_fan_in"]
max_gates_used_penalty_coeff = config["max_gates_used_penalty_coeff"]
if max_gates_used_penalty_coeff == 0:
    max_gates = jnp.array([0]*len(arch))
else:
    max_gates = jnp.array(config["max_gates"])
max_gates_used_penalty_coeff = max_gates_used_penalty_coeff / (sum(arch)-sum(max_gates))
continuous_penalty_coeff = config["continuous_penalty_coeff"]
min_gates_used_penalty_coeff = config["min_gates_used_penalty_coeff"]
if min_gates_used_penalty_coeff == 0:
    min_gates = jnp.array([0]*len(arch))
else:
    min_gates = jnp.array(config["min_gates"])
    min_gates_used_penalty_coeff = float(min_gates_used_penalty_coeff / (sum(min_gates)))
mean_fan_in_penalty_coeff = config["mean_fan_in_penalty_coeff"]
if mean_fan_in_penalty_coeff == 0:
    mean_fan_in = 0
else:
    mean_fan_in = config["mean_fan_in"]

dps = config["decimal_places"]

sig = jax.jit(jax.nn.sigmoid)
step = jax.jit(lambda x: jnp.where(x>0, 1, 0))

@partial(jax.jit, static_argnames="weight_activation")
def and_helper(
    x: jnp.ndarray,
    w: jnp.ndarray,
    weight_activation: Callable[[jnp.ndarray], jnp.ndarray]) -> float:
    """
    Helper function for forward, calculates the effective input a neuron
    receives from a specific previous layer (which is effectively a logical
    AND)

    Parameters
    x - could be inputs, could be outputs from a previous NAND gate,
    importantly it's a jnp array all from the same layer
    w - the weights of those wires connecting x to the NAND gate
    weight_activation - will be either sigmoid for the continuous version we
    use in training, or a step function for testing accuracy
    
    Returns
    the effective input from that layer for the NAND gate
    """
    return jnp.prod(1 + jnp.multiply(
        x, weight_activation(w)) - weight_activation(w))

and_cont = jax.jit(partial(and_helper, weight_activation=sig))
and_disc = jax.jit(partial(and_helper, weight_activation=step))

@partial(jax.jit, static_argnames="and_helper_func")
def forward(
    xs: jnp.ndarray, weights: jnp.ndarray,
    and_helper_func: Callable[[jnp.ndarray, jnp.ndarray], float]) -> float:
    """
    The forward pass for a neuron

    Parameters
    xs - a 2d jnp array of all the values on those wires
    weights - a 2d jnp array of all the wires going into it
    and_helper_func - function we use to compute the logical AND
    use in training, or a step function for testing accuracy
    
    Returns
    the continuous effective output for that NAND gate
    """
    return 1 - jnp.prod(jax.vmap(and_helper_func, in_axes=(0,0))(xs, weights))

forward_cont = jax.jit(partial(forward, and_helper_func=and_cont))
forward_disc = jax.jit(partial(forward, and_helper_func=and_disc))

@partial(jax.jit, static_argnames="layer_i")
def calc_surr(xs: jnp.ndarray, layer_i: int, surr_arr: List[jnp.ndarray]
              ) -> jnp.ndarray:
    """
    calculates the values of the surrogate NAND gates we're adding
    based on the current input
    Paramaters
    xs - the values calculated by the network thus far
    layer_i - the layer the surrogate NAND gates will be in
    surr_arr - the data structure telling us how to calculate the surrogate
    NAND gates

    Returns
    the surrogate NAND gates as a jnp array
    """
    start = [1-jnp.prod(
        xs[node[:,0], node[:,1]]) for node in surr_arr[layer_i]]
    return jnp.array(start)

@partial(jax.jit, static_argnames="forward_func")
def feed_forward(
    inputs: jnp.ndarray,
    neurons: jnp.ndarray,
    forward_func: Callable[[jnp.ndarray, jnp.ndarray], float],
    use_surr: bool=False,
    surr_arr: List[jnp.ndarray]=[]) -> jnp.ndarray:
    """
    Calculates the output of the network

    Parameters
    inputs - the input data
    neurons - the network
    forward_func - function used for calculating the next layer's output
    use in training, or a step function for testing accuracy
    use_surr - boolean value for if we're adding surrogate bits
    surr_arr - how to calculate the surrogate bits for if we're using them
    
    Returns
    the output of the network
    """
    xs = jnp.array([jnp.pad(
        inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])

    for layer_i in range(min(i_1-1, 3)):
        next = jax.vmap(forward_func, in_axes=(None, 0))(
            xs, neurons[layer_i])
        if use_surr and layer_i < len(surr_arr):
            next = jnp.concatenate([calc_surr(xs, layer_i), next])
        next = jnp.array([jnp.pad(
            next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])

    for layer_i in range(3, i_1-1):
        next = jax.vmap(forward_func, in_axes=(None, 0))(
            xs[jnp.array([0,-2,-1])], neurons[layer_i])
        if use_surr and layer_i < len(surr_arr):
            next = jnp.concatenate([calc_surr(xs, layer_i), next])
        next = jnp.array([jnp.pad(
            next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])

    return jax.vmap(forward_func, in_axes=(None, 0))(xs, neurons[i_1-1])[:outs]

feed_forward_cont = jax.jit(partial(feed_forward, forward_func=forward_cont))
feed_forward_disc = jax.jit(partial(feed_forward, forward_func=forward_disc))

@partial(jax.jit, static_argnames=('n', "and_helper_func"))
def forward_conv(
    xs: jnp.ndarray,
    weights:jnp.ndarray,
    s: int,
    n: int,
    and_helper_func: Callable[[jnp.ndarray, jnp.ndarray], float]) -> float:
    """
    Applies a filter of width `w` and stride `s` to the input array `xs`.
    
    Parameters:
    xs - an array of shape (old_channels, old_n, old_n), the input data
    weights - an array of shape (channels, old_channels, w, w), containing the
    filter weights
    s - the stride of the filter
    n - the new height and width of the picture
    and_helper_func - function we use to compute the logical AND

    Returns:
    An array of shape (channels, n, n), the result of applying the filter.
    """
    w = weights.shape[2]
    old_channels = xs.shape[0]
    channels = jnp.arange(weights.shape[0])
    return jax.vmap(
        lambda c: jax.vmap(
            lambda i: jax.vmap(
                lambda j: 1-and_helper_func(jax.lax.dynamic_slice(
                    xs,
                    (0, i*s, j*s),
                    (old_channels, w, w)), weights[c])
            )(jnp.arange(n))
        )(jnp.arange(n))
    )(channels)

forward_conv_cont = jax.jit(partial(forward_conv, and_helper_func=and_cont))
forward_conv_disc = jax.jit(partial(forward_conv, and_helper_func=and_disc))

@partial(jax.jit, static_argnames="forward_conv_func")
def feed_forward_conv(
    xs: jnp.ndarray,
    weights:jnp.ndarray,
    imgs_list: List[jnp.ndarray],
    convs: List[Tuple[int, int, int, int]],
    forward_conv_func: Callable[
        [jnp.ndarray, jnp.ndarray, int, int], jnp.ndarray]) -> jnp.ndarray:
    """
    Applies all of the convolutional layers to the input
    
    Parameters:
    xs - an array of shape (n, n), the input data
    weights - the list of weights
    weight_activation - will be either sigmoid for the continuous version we
    use in training, or a step function for testing accuracy
    
    Returns:
    The result of applying the convolutional layers, ready to be passed into
    the dense layers
    """
    for i, (ws, (_,_,s,n)) in enumerate(zip(weights, convs)):
        temp = forward_conv_func(xs, ws, s, n)
        xs = jnp.concatenate(
            [imgs_list[i], 1-imgs_list[i], temp, 1-temp], axis=0)
    return xs

feed_forward_conv_cont = jax.jit(partial(
    feed_forward_conv, forward_conv_func=forward_conv_cont, convs=convs))
feed_forward_conv_disc = jax.jit(partial(
    feed_forward_conv, forward_conv_func=forward_conv_disc, convs=convs))

print(convs)

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
        if add_img_or_custom == 'a':
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

def beta_sampler(shape: Tuple[int, ...], n: int, sigma: float, k: float=None
                 ) -> jnp.ndarray:
    """
    returns a set of numbers with the appropriate distribution. sigma must be
    at most sqrt(n-1)n this distribution ensures that the expected value of the
    sigmoid is 1/n
    
    Parameters
    shape - the shape of the array we want to return
    n - the number of wires into the NAND gate
    sigma - the standard deviation of the *beta* distribution

    Returns
    the weights
    """
    key = random.randint(0, 10000)
    alpha = ((n-1)/(n**2*sigma**2)-1)/n
    beta = alpha * (n - 1.0)
    samples = jax.random.beta(
        jax.random.key(key), a=alpha, b=beta, shape=shape)
    samples = jnp.clip(samples, epsilon, 1-epsilon)
    return jnp.log(samples / (1 - samples))

def normal_sampler1(shape: Tuple[int, ...], n: int, sigma: float, k: float=None
                    ) -> jnp.ndarray:
    """
    returns a set of numbers with the appropriate distribution.
    This distribution ensures that when we take n samples, we should expect one
    to be greater than 0
    
    Parameters
    shape - the shape of the array we want to return
    n - the number of wires into the NAND gate
    sigma - the standard deviation of the *beta* distribution

    Returns
    the weights
    """
    key = random.randint(0, 10000)
    mu = jsp_special.ndtri(1.0 / n)
    return sigma * jax.random.normal(jax.random.key(key), shape=shape) + mu

def normal_sampler2(shape: Tuple[int, ...], n: int, sigma: float, k: float
                    ) -> jnp.ndarray:
    """
    returns a set of numbers with the appropriate distribution.
    This distribution ensures that the expected value of the sigmoid is
    approximately 1/n
    
    Parameters
    shape - the shape of the array we want to return
    n - the number of wires into the NAND gate
    sigma - the standard deviation of the *beta* distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical
    relation isn't clear, so I pass it in separately)

    Returns
    the weights
    """
    key = random.randint(0, 10000)
    mu = -jnp.log(n-1)/k
    return sigma * jax.random.normal(jax.random.key(key), shape=shape) + mu

distribution_dict = {"beta_sampler": beta_sampler,
                     "normal_sampler1": normal_sampler1,
                     "normal_sampler2": normal_sampler2}
dense_distribution = distribution_dict[config["dense_distribution"]]
if add_img_or_custom == 'i':
    conv_distribution = distribution_dict[config["conv_distribution"]]

def get_weights_conv(
        w: int,
        c: int,
        old_c: int,
        distribution: Callable[[Tuple[int, ...], int, float, float], jnp.ndarray],
        sigma: jnp.ndarray,
        k: jnp.ndarray=None) -> jnp.ndarray:
    """
    Returns the weights for a filter

    Parameters
    w - the width of the filter
    c - the number of channels
    old_c - the number of channels in the previous layer
    distribution - one of the three functions defined above, takes a shape and
    some parameters
    sigma - the standard deviation of the normal distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical
    relation isn't clear, so I pass it in separately)
    
    Returns
    a 2d jnp array of the weights, which represents the wires going into a certain neuron
    """
    global key
    key = random.randint(0, 10000)
    n = old_c*w**2
    if k is None:
        return distribution(shape=(c, old_c, w, w), n=n, sigma=sigma, k=k)
    return normal_sampler2(shape=(c, old_c, w, w), n=n, sigma=sigma, k=k)

def initialise_conv(
        convs: List[Tuple[int, int, int, int]],
        distribution: Callable[[Tuple[int, ...], int, float, float], jnp.ndarray],
        sigma: jnp.ndarray,
        k: jnp.ndarray=None) -> Network:
    """
    creates the weights data structure for the convolutional layers

    Parameters
    convs - a tuple describing the convolutional layers we're adding
    distribution - one of the three functions defined above, takes a shape and
    some parameters
    sigma - the standard deviation of the normal distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical
    relation isn't clear, so I pass it in separately)
    
    Returns
    the convolutional layers of the network
    """
    neurons = []
    current_c = 2
    for w,_,c,_ in convs:
        weights = get_weights_conv(
            w, c//2-1, current_c, distribution, sigma, k)
        neurons.append(weights)
        current_c = c
    return tuple(neurons)

def get_weights(
        layer: int,
        arch: List[int],
        distribution: Callable[[Tuple[int, ...], int, float, float], jnp.ndarray],
        sigma: jnp.ndarray,
        k: jnp.ndarray=0.) -> jnp.ndarray:
    """
    Returns the weights of a given neuron

    Parameters
    layer - the layer that the neuron is in
    arch - a list representing the architecture
    distribution - one of the three functions defined above, takes a shape and
    some parameters
    sigma - the standard deviation of the normal distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical
    relation isn't clear, so I pass it in separately)
    
    Returns
    a 2d jnp array of the weights, which represents the wires going into a
    certain neuron
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
    if layer == 1 or layer == 2 or layer == len(arch)-1:
        for i in range(layer):
            inner_layer = distribution(shape=(arch[i],), n=n, sigma=sigma, k=k)
            weights = weights.at[i].set(jnp.pad(
                inner_layer,
                (0, i_4-arch[i]),
                mode="constant", constant_values=-jnp.inf))
            key = random.randint(0, 10000)
    else:
        inner_layer = distribution(shape=(arch[0],), n=n, sigma=sigma, k=k)
        weights = weights.at[0].set(jnp.pad(
            inner_layer,
            (0, i_4-arch[0]),
            mode="constant", constant_values=-jnp.inf))
        key = random.randint(0, 10000)
        for i in range(1,3):
            inner_layer = distribution(
                shape=(arch[layer-3+i],), n=n, sigma=sigma, k=k)
            weights = weights.at[i].set(jnp.pad(
                inner_layer,
                (0, i_4-arch[layer-3+i]),
                mode="constant", constant_values=-jnp.inf))
            key = random.randint(0, 10000)
    return weights

def initialise(
        arch: List[int],
        true_arch: List[int],
        distribution: Callable[[Tuple[int, ...], int, float, float], jnp.ndarray],
        sigma: jnp.ndarray,
        k: jnp.ndarray=0.) -> List[jnp.ndarray]:
    """
    initialises the network

    Parameters
    arch - a list representing the architecture
    distribution - one of the three functions defined above, takes a shape and
    some parameters
    sigma - the standard deviation of the normal distribution
    k - a rescaling factor (it's dependent on sigma, but the mathematical
    relation isn't clear, so I pass it in separately)
    
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
            layer = layer.at[i2].set(get_weights(
                i1, true_arch, distribution, sigma, k))
        neurons.append(layer)
    return tuple(neurons)

@jax.jit
def max_fan_in_penalty_disc(neurons: Network, max_fan_in: int) -> float:
    """
    calculates a penalty, which is minimised for any maximum fan-in under or
    equal to "max_fan_in"
    this doesn't account for duplicate gates, and this is the discrete version.

    Parameters
    neurons - the network
    max_fan_in - the desired maximum fan-in
    
    Returns
    the penalty
    """
    fan_ins = jnp.array([])
    for layer in neurons:
        fan_ins = jnp.concatenate((fan_ins, jax.vmap(
            lambda x:jnp.sum(jnp.where(x>0, 1, 0)))(layer)))
    temp = jax.nn.relu(fan_ins-max_fan_in)
    return jnp.max(temp)

@jax.jit
def max_fan_in_penalty(neurons: Network, max_fan_in: int, temperature: float
                       ) -> float:
    """
    calculates a penalty, which is minimised for any maximum fan-in under or
    equal to "max_fan_in". This doesn't account for duplicate gates

    Parameters
    neurons - the network
    max_fan_in - the desired maximum fan-in
    temperature - lower makes it closer to discrete
    
    Returns
    the penalty (a float, which will be multiplied by a coefficient, and added
    to the loss)
    """
    fan_ins = jnp.array([])
    for layer in neurons:
        fan_ins = jnp.concatenate((fan_ins, jax.vmap(
            lambda x:jnp.sum(sig(x/temperature)))(layer)))
    temp = jax.nn.relu(fan_ins-max_fan_in)
    return jnp.sum(jax.nn.softmax(temp)*temp)

@jax.jit
def mean_fan_in_penalty(
    neurons: Network,
    mean_fan_in: float,
    temperature: float,
    num_neurons: int) -> float:
    """
    calculates a penalty, which is minimised for any mean fan-in under or equal
    to "mean_fan_in". This doesn't account for duplicate gates

    Parameters
    neurons - the network
    mean_fan_in - the desired mean fan-in
    temperature - lower makes it closer to discrete
    num_neurons - the number of neurons in the network
    
    Returns
    the penalty (a float, whihc will be multiplied by a coefficient, and added
    to the loss)
    """
    fan_ins = jnp.array([])
    for layer in neurons:
        fan_ins = jnp.concatenate((fan_ins, jax.vmap(
            lambda x:jnp.sum(sig(x/temperature)))(layer)))
    temp = jnp.sum(fan_ins)/num_neurons
    return jax.nn.relu(temp-mean_fan_in)

@partial(jax.jit, static_argnums=1)
def cont_or_arr(arr: jnp.ndarray, axis=None) -> jnp.ndarray:
    # computes a continuous or of a jnp array over the specified axis,
    # using De Morgan's
    return 1-jnp.prod(1-arr, axis=axis)

@jax.jit
def cont_or(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    # computes the element-wise continuous or of the inputs
    return 1-(1-arr1)*(1-arr2)

@partial(jax.jit, static_argnums=0)
def input_layers(layer: int) -> jnp.ndarray:
    # returns the range of layers that go into the input layer number
    if layer <= 2 or layer == len(arch) - 1:
        return jnp.arange(layer)
    return jnp.array([0, layer-2, layer-1])

@partial(jax.jit, static_argnames="weight_activation")
def get_used_array(neurons: Network, weight_activation: Callable[[jnp.ndarray], jnp.ndarray]) -> float:
    """
    returns an array, used, representing the network, where if
    used[layer][i] is close to 1, the neuron is used.
    used_back tells us if the NAND gates will be used in the outputs
    used_for tells us if the NAND gates are connected to the inputs
    and so their product tells us if the NAND gates are being used
    the doesn't account for duplicate gates

    Parameters
    neurons - the network
    weight_activation - sigmoid with temperature or step function
    
    Returns
    the array
    """
    sig_neurons = [weight_activation(layer) for layer in neurons]
    used_back = jnp.zeros(shape=(len(arch), i_4))
    used_back = used_back.at[len(arch)-1, :outs].set(jnp.ones(shape=outs))
    # outputs are used by outputs
    for layer in range(len(arch)-1, 0, -1):
        temp = (sig_neurons[layer-1]
                * used_back[layer, :arch[layer]][:, jnp.newaxis, jnp.newaxis])
        # this is a 2D matrix, the LHS of the * is how much each neuron to the
        # left of this neuron is used by this neuron. The RHS of the * is a
        # vector, which is how much this neuron is used by the output.
        temp = cont_or_arr(temp, axis=0)
        used_back = used_back.at[input_layers(layer)].set(
            cont_or(used_back[input_layers(layer)], temp))
    used_for = jnp.zeros(shape=(len(arch), i_4))
    used_for = used_for.at[0, :new_ins].set(jnp.ones(shape=new_ins))
    for layer in range(1, len(arch)):
        temp = (sig_neurons[layer-1][:arch[layer]]
                * used_for[input_layers(layer)][jnp.newaxis,:,:])
        temp = cont_or_arr(temp, axis=(1,2))
        used_for = used_for.at[layer, :arch[layer]].set(
            cont_or(used_for[layer, :arch[layer]], temp))
    return used_back*used_for

@jax.jit
def max_gates_used_penalty(neurons: Network, max_gates: jnp.ndarray) -> float:
    """
    calculates a penalty, which is maximised for any gate usage less than or
    equal to "max_gates" this doesn't account for duplicate gates.

    Parameters
    neurons - the network
    max_gates - an array specifying the max number of nodes in each layer.
    weight_activation - sigmoid with a temperature or step function
    
    Returns
    the penalty (a float, which will be multiplied by a coefficient, and added
    to the loss)
    """
    used = get_used_array(neurons, lambda x: sig(x/temperature))
    return jnp.sum(jax.nn.relu(jnp.sum(used, axis=1)-max_gates))

@jax.jit
def min_gates_used_penalty(neurons: Network, min_gates: jnp.ndarray) -> float:
    """
    calculates a penalty, which is maximised for any gate usage greater than or
    equal to "min_gates". This doesn't account for duplicate gates.

    Parameters
    neurons - the network
    min_gates - an array specifying the min number of nodes in each layer.
    weight_activation - sigmoid with a temperature or step function
    
    Returns
    the penalty (a float, which will be multiplied by a coefficient, and added
    to the loss)
    """
    used = get_used_array(neurons, lambda x: sig(x/temperature))
    return jnp.sum(jax.nn.relu(min_gates-jnp.sum(used, axis=1)))

@partial(jax.jit, static_argnames="weight_activation")
def gate_usage_by_layer(neurons: Network, weight_activation: Callable[[jnp.ndarray], jnp.ndarray]
                        ) -> float:
    # gives us the gate usage by layer
    return jnp.sum(get_used_array(neurons, weight_activation), axis=1)

@jax.jit
def continuous_penalty(neurons: Network, num_wires: int) -> float:
    """
    calculates a penalty which is minimised when the weights have a high
    magnitude. Adding this to the loss can lead to networks where a low loss is
    more strongly correlated with a high accuracy

    Parameters
    neurons - the network
    total - the number of weights in the network
    
    Returns
    the penalty (a float, which will be multiplied by a coefficient, and added
    to the loss)
    """
    s = sum([jnp.sum(
        1-sig(jnp.absolute(layer))) for layer in neurons])
    return s/num_wires

epsilon = 1e-7
@jax.jit
def bce_loss(
    neurons: Network,
    inputs: jnp.ndarray,
    output: jnp.ndarray,
    mask1: jnp.ndarray=None,
    mask2:jnp.ndarray=None,
    use_surr: bool=False,
    surr_arr: List[jnp.ndarray]=[]) -> float:
    """
    calculates the binary cross entropy loss

    Parameters
    neurons - the network
    inputs - all of the inputs (training xs)
    output - all of the outputs (training labels or ys)
    mask1 - a mask for samples we got right
    mask2 - a mask for the samples we got wrong
    
    Returns
    loss
    """
    pred = jax.vmap(feed_forward_cont, in_axes=(0, None, None, None, None))(
        inputs, neurons, sig, use_surr, surr_arr)
    pred = jnp.clip(pred, epsilon, 1-epsilon)
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    if mask1 != None:
        # separately calculating the loss of the correct and incorrect outputs.
        # Weighs the rarer one (which should be incorrect) more heavily. The
        # idea is to give it a bigger nudge to overcome a small final hurdle.
        l11 = jnp.mean(optax.sigmoid_binary_cross_entropy(
            pred_logits[mask1], output[mask1]))
        l12 = jnp.mean(optax.sigmoid_binary_cross_entropy(
            pred_logits[mask2], output[mask2]))
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
        return (c1*l11 + c2*l12)/2
    else:
        return jnp.mean(
            optax.sigmoid_binary_cross_entropy(pred_logits, output))
    
epsilon = 1e-7
@jax.jit
def loss(
    neurons: Network,
    inputs: jnp.ndarray,
    output: jnp.ndarray,
    mask1: jnp.ndarray=None,
    mask2:jnp.ndarray=None,
    use_surr: bool=False,
    surr_arr: List[jnp.ndarray]=[],
    max_fan_in: int=None,
    temperature: float=None,
    mean_fan_in: float=None,
    max_gates: List[int]=None,
    min_gates: List[int]=None,
    num_neurons: int=None,
    num_wires: int=None) -> float:
    """
    calculates the loss

    Parameters
    neurons - the network
    inputs - all of the inputs (training xs)
    output - all of the outputs (training labels or ys)
    mask1 - a mask for samples we got right
    mask2 - a mask for the samples we got wrong
    
    Returns
    loss
    """
    l = bce_loss(neurons, inputs, output, mask1, mask2, use_surr, surr_arr)
    if max_fan_in_penalty_coeff:
        l += (max_fan_in_penalty_coeff
              * max_fan_in_penalty(neurons, max_fan_in, temperature))
    if mean_fan_in_penalty_coeff:
        l += (mean_fan_in_penalty_coeff
              * mean_fan_in_penalty(
                  neurons, mean_fan_in, temperature, num_neurons))
    if max_gates_used_penalty_coeff:
        l += (max_gates_used_penalty_coeff
              * max_gates_used_penalty(neurons, max_gates))
    if min_gates_used_penalty_coeff:
        l += (min_gates_used_penalty_coeff
              * min_gates_used_penalty(neurons, min_gates))
    if continuous_penalty_coeff:
        l += continuous_penalty_coeff * continuous_penalty(neurons, num_wires)
    return l

grad = jax.jit(jax.grad(loss, argnums=0))

@jax.jit
def loss_conv(
    network: List[Network],
    inputs: jnp.ndarray,
    output: jnp.ndarray,
    scaled: List[jnp.ndarray]=None,
    convs: List[Tuple[int, int, int, int]]=None,
    max_fan_in: int=None,
    temperature: float=None,
    mean_fan_in: float=None,
    max_gates: List[int]=None,
    min_gates: List[int]=None,
    num_neurons: int=None,
    num_wires: int=None) -> float:
    """
    calculates the loss including convolutional layers

    Parameters
    network - [neurons, neurons_conv], where neurons are the dense layers, and
    neurons_conv are the convolutional
    inputs - all of the inputs (training xs)
    output - all of the outputs (training labels or ys)
    
    Returns
    loss
    """
    if convs is None:
        inputs = inputs.reshape(inputs.shape[0], -1)
        return loss(network[0], inputs, output, max_fan_in=max_fan_in,
                    temperature=temperature, mean_fan_in=mean_fan_in,
                    max_gates=max_gates, min_gates=min_gates,
                    num_neurons=num_neurons, num_wires=num_wires)
    pred = jax.vmap(feed_forward_conv_cont, in_axes=(0, None, 0))(
        inputs, network[1], scaled)
    pred = pred.reshape(pred.shape[0], -1)
    return loss(network[0], pred, output, max_fan_in=max_fan_in,
                temperature=temperature, mean_fan_in=mean_fan_in,
                max_gates=max_gates, min_gates=min_gates,
                num_neurons=num_neurons, num_wires=num_wires)

grad_conv = jax.jit(jax.grad(loss_conv, argnums=0))

@jax.jit
def test(neurons: Network,
         inputs: jnp.ndarray,
         output: jnp.ndarray,
         use_surr: bool=False,
         surr_arr: List[jnp.ndarray]=[]) -> bool:
    """
    is true iff the network is 100% accurate

    Parameters
    neurons - the network
    inputs - jnp array of the inputs we're testing
    output - jnp array of the outputs we're testing
    use_surr - boolean telling us if we're using surrogate bits
    surr_arr - data structure of how to calculate surrogate bits
    
    Returns
    if the network was 100% accurate
    """
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None, None, None, None))(
        inputs, neurons, step, use_surr, surr_arr)
    return jnp.all(pred==output)

current_max_fan_in = -1
def test_fan_in(neurons: Network) -> bool:
    """
    is true iff the max fan-in is less than what the user specified (ignoring
    duplicates)

    Parameters
    neurons - the network
    
    Returns
    if the max fan-in is less than what the user specified
    """
    global current_max_fan_in
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

@partial(jax.jit, static_argnames="skew_towards_falses")
def acc(neurons: Network,
        inputs: jnp.ndarray,
        output: jnp.ndarray,
        use_surr: bool=False,
        surr_arr: List[jnp.ndarray]=[],
        skew_towards_falses=False
        ) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    calculates the accuracy, and also the masks used in the loss function

    Parameters
    neurons - the network
    inputs - jnp array of the inputs we're testing
    output - jnp array of the outputs we're testing
    use_surr - boolean telling us if we're using surrogate bits
    surr_arr - data structure of how to calculate surrogate bits
    skew_towards_false - boolean telling us if we're gonna need to calculate
    the masks (which we use to bias the gradients more towards what it's
    getting wrong)
    
    Returns
    accuracy - the accuracy (may be specifically the testing accuracy)
    mask1 - the mask of the samples it got right
    mask2 - the mask of the samples it got wrong
    """
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None, None, None, None))(
        inputs, neurons, step, use_surr, surr_arr)
    pred = (pred == output)
    pred = jnp.sum(pred, axis=1)
    if skew_towards_falses:
        trues = jnp.where(pred == outs)
        falses = jnp.where(pred < outs)
        return jnp.sum(pred)/((2**(ins))*(outs)), trues[0], falses[0]
    return jnp.sum(pred)/((2**(ins))*(outs)), None, None

@jax.jit
def acc_conv(network: List[Network],
             inputs: jnp.ndarray,
             output: jnp.ndarray,
             scaled: List[jnp.ndarray]=None,
             convs: List[Tuple[int, int, int, int]]=None
             ) -> float:
    """
    calculates the accuracy for images

    Parameters
    network - [neurons, neurons_conv], where neurons are the dense layers, and
    neurons_conv are the convolutional
    inputs - jnp array of the inputs we're testing
    output - jnp array of the outputs we're testing
    
    Returns
    accuracy - the accuracy (may be specifically the testing accuracy)
    """
    if not (convs is None):
        inputs = jax.vmap(feed_forward_conv_disc, in_axes=(0, None, 0))(
            inputs, network[1], scaled)
    inputs = inputs.reshape(inputs.shape[0], -1)
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None, None))(
        inputs, network[0])
    result = jax.vmap(image_util.evaluate)(pred, output)
    return jnp.sum(result)/result.size

batches = config["batches"]
batch_size = num_ins//batches

boundary_jump = 5*(max(10//batches,1)**2)*batch_size
lr_multiplier = batch_size**0.5

schedule_dense = optax.join_schedules(
    schedules = [optax.constant_schedule(
        lr*lr_multiplier) for lr in config["lr_dense"]],
    # schedules=[
    #     optax.constant_schedule(1.0*lr_multiplier),
    #     optax.constant_schedule(0.1*lr_multiplier),
    #     optax.constant_schedule(0.03*lr_multiplier),
    #     optax.constant_schedule(0.01*lr_multiplier),
    #     optax.constant_schedule(0.003*lr_multiplier),
    #     optax.constant_schedule(0.001*lr_multiplier),
    # ],
    boundaries=[(i+1)**2*boundary_jump for i in range(1)]
)

optimizer_dense = optax.adam(learning_rate=schedule_dense)

if convs:
    schedule_conv = optax.join_schedules(
        schedules = [optax.constant_schedule(
            lr*lr_multiplier) for lr in config["lr_conv"]],
        # schedules=[
        #     optax.constant_schedule(1.0*lr_multiplier),
        #     optax.constant_schedule(0.1*lr_multiplier),
        #     optax.constant_schedule(0.03*lr_multiplier),
        #     optax.constant_schedule(0.01*lr_multiplier),
        #     optax.constant_schedule(0.003*lr_multiplier),
        #     optax.constant_schedule(0.001*lr_multiplier),
        # ],
        boundaries=[(i+1)**2*boundary_jump for i in range(1)]
    )

    optimizer_conv = optax.adam(learning_rate=schedule_conv)

print("Learning:\n", output, "\nwith arch:", true_arch)
start_time = time.time()
neurons = initialise(arch, true_arch, dense_distribution, dense_sigma, dense_k)
if add_img_or_custom == 'i':
    neurons_conv = initialise_conv(convs, conv_distribution, conv_sigma, conv_k)
    if convs:
        opt_state_conv = optimizer_conv.init(neurons_conv)
else:
    opt_state_dense = optimizer_dense.init(neurons)
init_time = time.time()
print("Took", init_time-start_time, "seconds to initialise.")
print([layer.shape for layer in neurons])
if add_img_or_custom == 'i' and convs:
    print([layer.shape for layer in neurons_conv])

# @jax.jit
def batch_comp(func: Callable, batch_size: int, batches: int, *args, **kwargs
               ) -> List:
    """
    Takes a function and its arguments, and applies it in batches, returning a
    list of the results. If th function takes some arguments that aren't
    meant to be batched, then can instead pass in a lambda or a partial, with
    those arguments pre-applied.

    Parameters
    func - the function we're computing in batches
    batch_size - the number of elements in each batch
    batches - the number of batches
    
    Returns
    a list of the outputs computed for each batch
    """
    output = 0
    for batch_number in range(batches):
        sliced_args = tuple(arg[
            batch_number*batch_size:(batch_number+1)*batch_size
            ] for arg in args)
        sliced_kwargs = {k:[imgs[batch_number*batch_size:(
            batch_number+1)*batch_size] for imgs in v] if k=="scaled" else v[
            batch_number*batch_size:(batch_number+1)*batch_size
            ] for k, v in kwargs.items()}
        output += func(*sliced_args, **sliced_kwargs)
    return output/batches

loss_kwargs = {"max_fan_in": max_fan_in,
               "temperature": temperature,
               "mean_fan_in": mean_fan_in,
               "max_gates": max_gates,
               "min_gates": min_gates,
               "num_neurons": num_neurons,
               "num_wires": num_wires,
               "use_surr": use_surr,
               "surr_arr": surr_arr}

loss_conv_kwargs = {"max_fan_in": max_fan_in,
                    "temperature": temperature,
                    "mean_fan_in": mean_fan_in,
                    "max_gates": max_gates,
                    "min_gates": min_gates,
                    "num_neurons": num_neurons,
                    "num_wires": num_wires,}

if add_img_or_custom == 'i':
    accuracy = batch_comp(
        partial(acc_conv, network=[neurons, neurons_conv], convs=convs),
        batch_size, x_test.shape[0]//batch_size,
        inputs=x_test, output=y_test, scaled=scaled_test_imgs)
    new_loss = batch_comp(
        partial(loss_conv, network=[neurons, neurons_conv], convs=convs, **loss_conv_kwargs),
        batch_size, batches,
        inputs, output, scaled=scaled_train_imgs)
    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}")
    print(gate_usage_by_layer(neurons, sig))
    print(gate_usage_by_layer(neurons, step))
    print(max_fan_in_penalty(neurons, 0, temperature), max_fan_in_penalty_disc(neurons, 0))
    print(mean_fan_in_penalty(neurons, 0, temperature, num_neurons))
else:
    accuracy = batch_comp(
        partial(acc, neurons=neurons, use_surr=use_surr, surr_arr=surr_arr),
        batch_size, batches,
        inputs, output, False)
    new_loss = batch_comp(
        partial(loss, network=[neurons, neurons_conv], **loss_kwargs),
        batch_size, batches,
        inputs, output)
    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}")
    print(gate_usage_by_layer(neurons, sig))
    print(gate_usage_by_layer(neurons, step))
    print(max_fan_in_penalty(neurons, 0, temperature), max_fan_in_penalty_disc(neurons, 0))
    print(mean_fan_in_penalty(neurons, 0, temperature, num_neurons))

def run(timeout=config["timeout"]) -> None:
    global inputs, output, neurons, neurons_conv, opt_state_dense, opt_state_conv, scaled_train_imgs
    cont = True
    iters = 0
    file_i = -1
    start_run_time = time.time()
    while cont:
        iters += 1
        for _ in range(max(10//batches, 1)):
            if batches > 1:
                key = jax.random.PRNGKey(random.randint(0, 10000))
                shuffled_indices = jax.random.permutation(key, inputs.shape[0])
                inputs = inputs[shuffled_indices]
                output = output[shuffled_indices]
                if add_img_or_custom == 'i' and convs:
                    scaled_train_imgs = [imgs[shuffled_indices] for imgs in scaled_train_imgs]
            for batch in range(batches):
                if add_img_or_custom == 'i':
                    gradients = grad_conv([neurons, neurons_conv],
                                        inputs[batch*batch_size:(batch+1)*batch_size],
                                        output[batch*batch_size:(batch+1)*batch_size],
                                        [imgs[batch*batch_size:(batch+1)*batch_size] for imgs in scaled_train_imgs],
                                        convs, **loss_conv_kwargs)
                    update, opt_state_dense = optimizer_dense.update(gradients[0], opt_state_dense, neurons)
                    neurons = optax.apply_updates(neurons, update)
                    if convs:
                        update, opt_state_conv = optimizer_conv.update(gradients[1], opt_state_conv, neurons_conv)
                        neurons_conv = optax.apply_updates(neurons_conv, update)
                else:
                    gradients = grad(neurons,
                                    inputs[batch*batch_size:(batch+1)*batch_size],
                                    output[batch*batch_size:(batch+1)*batch_size],
                                    **loss_kwargs)
                    updates, opt_state_dense = optimizer_dense.update(gradients, opt_state_dense, neurons)
                    neurons = optax.apply_updates(neurons, updates)
            if time.time() - start_run_time > timeout * 60:
                if add_img_or_custom == 'i':
                    accuracy = batch_comp(
                        partial(acc_conv, network=[neurons, neurons_conv], convs=convs),
                        batch_size, x_test.shape[0]//batch_size,
                        inputs=x_test, output=y_test, scaled=scaled_test_imgs)
                    new_loss = batch_comp(
                        partial(loss_conv, network=[neurons, neurons_conv], convs=convs, **loss_conv_kwargs),
                        batch_size, batches,
                        inputs, output, scaled=scaled_train_imgs)
                    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}")
                    print(gate_usage_by_layer(neurons, sig))
                    print(gate_usage_by_layer(neurons, step))
                    print(max_fan_in_penalty(neurons, 0, temperature), max_fan_in_penalty_disc(neurons, 0))
                    print(mean_fan_in_penalty(neurons, 0, temperature, num_neurons))
                else:
                    accuracy = batch_comp(
                        partial(acc, neurons=neurons, use_surr=use_surr, surr_arr=surr_arr),
                        batch_size, batches,
                        inputs, output, False)
                    new_loss = batch_comp(
                        partial(loss, network=[neurons, neurons_conv], **loss_kwargs),
                        batch_size, batches,
                        inputs, output)
                    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}")
                    print(gate_usage_by_layer(neurons, sig))
                    print(gate_usage_by_layer(neurons, step))
                    print(max_fan_in_penalty(neurons, 0, temperature), max_fan_in_penalty_disc(neurons, 0))
                    print(mean_fan_in_penalty(neurons, 0, temperature, num_neurons))
                return
        if add_img_or_custom != 'i':
            if test(neurons) and (max_fan_in_penalty_coeff==0 or test_fan_in(neurons)) or get_optional_input_non_blocking() == 2:
                cont = False
        if cont:
            if iters == max(10//batches, 1):
                if add_img_or_custom == 'i':
                    accuracy = batch_comp(
                        partial(acc_conv, network=[neurons, neurons_conv], convs=convs),
                        batch_size, x_test.shape[0]//batch_size,
                        inputs=x_test, output=y_test, scaled=scaled_test_imgs)
                    new_loss = batch_comp(
                        partial(loss_conv, network=[neurons, neurons_conv], convs=convs, **loss_conv_kwargs),
                        batch_size, batches,
                        inputs, output, scaled=scaled_train_imgs)
                    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}")
                    print(gate_usage_by_layer(neurons, sig))
                    print(gate_usage_by_layer(neurons, step))
                    print(max_fan_in_penalty(neurons, 0, temperature), max_fan_in_penalty_disc(neurons, 0))
                    print(mean_fan_in_penalty(neurons, 0, temperature, num_neurons))
                else:
                    accuracy = batch_comp(
                        partial(acc, neurons=neurons, use_surr=use_surr, surr_arr=surr_arr),
                        batch_size, batches,
                        inputs, output, False)
                    new_loss = batch_comp(
                        partial(loss, network=[neurons, neurons_conv], **loss_kwargs),
                        batch_size, batches,
                        inputs, output)
                    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}")
                    print(gate_usage_by_layer(neurons, sig))
                    print(gate_usage_by_layer(neurons, step))
                    print(max_fan_in_penalty(neurons, 0, temperature), max_fan_in_penalty_disc(neurons, 0))
                    print(mean_fan_in_penalty(neurons, 0, temperature, num_neurons))
                iters = 0
    end_time = time.time()
    print("Took", end_time-start_run_time, "seconds to train.")
    if add_img_or_custom != 'i':
        circuit = output_circuit(neurons, True, True)
        [print(circ) for circ in circuit]
    return

run()
