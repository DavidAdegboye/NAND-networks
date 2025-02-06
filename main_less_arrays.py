import jax
import jax.numpy as jnp
import optax
import random
import itertools
from typing import List, Tuple, Set, Union
import time

import sys
import os
if os.name == 'nt':  # Windows
    import msvcrt
else:  # Unix-like systems
    import select

new_batches = 0
def get_optional_input_non_blocking():
    global new_batches
    if os.name == 'nt':  # Windows
        if msvcrt.kbhit():
            user_input = msvcrt.getch().decode('utf-8').strip()
            if 's' in user_input:
                return 1
            if 'd' in user_input:
                return 2
            if 'b' in user_input:
                return 3
            if user_input.isnumeric():
                new_batches = int(user_input)
                return 4
    else:  # Unix-like systems
        input_ready, _, _ = select.select([sys.stdin], [], [], 0)  # Non-blocking select
        if input_ready:
            user_input = sys.stdin.readline().strip()
            if 's' in user_input:
                return 1
            if 'd' in user_input:
                return 2
            if 'b' in user_input:
                return 3
            if user_input.isnumeric():
                new_batches = int(user_input)
                return 4
    return 0

# defining some types
Network = List[jnp.ndarray]
NeuronShape = List[int]
LayerShape = List[NeuronShape]
NetworkShape = List[LayerShape]
##jax.config.update("jax_traceback_filtering", "off")

print(jax.devices())

add_or_img = input("Do you want to learn a custom boolean circuit(c), an adder(a) or an image classifier(i)?\n")

if add_or_img == 'c':
    import custom_util
    inputs, output, ins, outs, num_ins = custom_util.set_up_custom()
elif add_or_img == 'a':
    import adders_util
    inputs, output, ins, outs, num_ins = adders_util.set_up_adders()
else:
    import image_class
    inputs, x_test, output, y_test, num_ins = image_class.set_up_img()
    inputs = jnp.expand_dims(inputs, axis=1)
    x_test = jnp.expand_dims(x_test, axis=1)
    outs = output.shape[1]

def add_second_layers(input: jnp.ndarray, min_fan: int, max_fan: int) -> jnp.ndarray:
    """
    adds extra bits to the input to help aid in training. These extra bits are produced by putting the current input into
    NAND gates.

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
    convs, true_arch = image_class.add_real_conv()
    if convs:
        new_ins = convs[-1][2] * convs[-1][3]**2
    else:
        new_ins = true_arch[0]
    add_comp = input("Add a complement layer? yes(y) or no(n)\n")
    if add_comp == 'y':
        new_ins *= 2
else:
    true_arch = []
    # for adders and arbitrary combinational logic circuits, we're first adding extra layers
    extra_layers = []
    add_extra = input("Add extra layer? Yes(y) or no(n)\n")
    while add_extra == 'y':
        min_fan = int(input("Min fan-in of this layer:\n"))
        min_fan = max(min_fan, 1)
        max_fan = int(input("Max fan-in of this layer:\n"))
        max_fan = min(max_fan, inputs.shape[1])
        extra_layers.append((min_fan, max_fan))
        old_ins = inputs.shape[1]
        inputs = jax.vmap(add_second_layers, in_axes=(0, None, None))(inputs, min_fan, max_fan)
        if add_or_img == 'i':
            x_test = jax.vmap(add_second_layers, in_axes=(0, None, None))(x_test, min_fan, max_fan)
        else:
            mask = jnp.sum(inputs, axis=0) < 2**ins
            inputs = inputs[:, mask]
        print(inputs)
        new_ins = inputs.shape[1]
        true_arch.append(new_ins - old_ins)
        add_extra = input("Add another extra layer? Yes(y) or no(n)\n")
    # and then if it's an adder, we're also adding extra help for adders
    if add_or_img == 'a':
        inputs, true_arch, add_adder_help, with_nots = adders_util.adder_help(inputs, true_arch)
    new_ins = inputs.shape[1]
print(true_arch)

# I found using the same distribution for weights globally, and a sigma value of 1.5 works best, particularly for large networks
# which is what I'm setting up here for images. Still however adding the option for fine tuning for the other ones.
if add_or_img == 'i':
    global_weights = 'g'
else:
    global_weights = input("Global weights(g) or local(l)?\n")
    sigma_i = int(input("Which index for the sigma (recommend 4 for local, 10 for global)?:\n"))

# I've found linear works the best for adders, although there may be a different way to taper down I've not tried.
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
    if add_or_img != 'i':
        wide = input("Add width(w) or depth(d)?\n")
    arch = [new_ins] + [width] * hidden + [outs]
else:
    starting_width = int(input("Starting Width:\n"))
    hidden = int(input("No. hidden layers:\n"))
    if add_or_img != 'i':
        wide = input("Add width(w) or depth(d)?\n")
    diff = starting_width - outs
    layer_diff = diff/hidden
    arch = [new_ins] + [round(starting_width-i*layer_diff) for i in range(hidden)] + [outs]

l2_coeff = float(input("l2 coefficient:\n"))
max_fan_in = int(input("What should the max fan-in of the whole network be?:\n"))
l3_coeff = float(input("l3 coefficient:\n"))
max_gates = int(input("What should the max number of gates in network be?:\n"))

# for adders and arbitrary combinational logic circuits, where we're aiming for 100% accuracy, if we're stuck
# in the high nineties at a local minima, I've added this to give a little nudge. It makes the losses of the
# incorrect samples weigh more.
weigh_even = 'n'

batches = num_ins
# batches = int(input("How many batches?\n"))
batch_size = num_ins//batches

@jax.jit
def f(x: jnp.ndarray, w: jnp.ndarray) -> float:
    """
    More of a helper function for forward, but it calculates the continuous effective input a neuron receives from a specific previous layer

    Parameters
    x - could be inputs, could be outputs from a previous NAND gate, importantly it's a 1D jnp array all from the same layer
    w - the weights of those wires connecting x to the NAND gate
    
    Returns
    the continuous effective input from that layer for the NAND gate
    """
    # x would be all of the inputs coming in from a certain layer
    # w would be all of the weights for inputs to that layer to a given NAND gate
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w))

@jax.jit
def f_disc(x: jnp.ndarray, w: jnp.ndarray) -> int:
    """
    More of a helper function for forward_disc, but it calculates the discrete effective input a neuron receives from a specific previous layer

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
    weights - a 2d jnp array of all the wires going into it
    xs - a 2d jnp array of all the values on those wires
    
    Returns
    the continuous effective output for that NAND gate
    """
    # the forward pass for an arbitrary neuron. 1 - the product of all the fs
    # to use vmap, I include some padding that doesn't affect the value.
    # x=1, w=0, since f(1,0)=1, so it wouldn't affect the result
    # after the product.
    return 1 - jnp.prod(jax.vmap(f)(xs, weights))

@jax.jit
def forward_disc(xs: jnp.ndarray, weights: jnp.ndarray) -> int:
    """
    The discrete forward pass for a neuron

    Parameters
    weights - a 2d jnp array of all the wires going into it
    xs - a 2d jnp array of all the values on those wires
    
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
    # if we've added some helper bits, we're unpacking them into their separate layers
    if true_arch:
        eff_arch = arch[1:].copy()
        eff_arch = [ins] + true_arch + eff_arch
    else:
        eff_arch = arch.copy()
    if verbose:
        print(eff_arch)
    current_h = eff_arch[0]
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
                current_h = current_l + eff_arch[layer_i]
            current += 1
    output.append(current)
    output += [0] * (len(output) - len(eff_arch))
    output.append(outs)
    return output

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
    queue = list(used)
    nodes = []
    while len(queue):
        node_i = queue.pop(0)
        nodes.append(node_i)
        for node_2 in connecteds[node_i]:
            if node_2 not in used:
                queue.append(node_2)
                used.add(node_2)
    # print(nodes)
    used_list: List[int] = sorted(list(used))
    if verbose:
        print(used_list)
    learnt_arch = get_used(used_list, arch, verbose)
    fan_ins = []
    for node_index in used_list:
        if node_index >= learnt_arch[0]:
            fan_ins.append(len(connecteds[node_index]))
    with open(f"circuit.txt", "w") as f:
        f.write(f"used:\n{learnt_arch}\nout of:\n{arch}\n")
        f.write(f"Max fan-in: {max(fan_ins)}\nAverage fan-in: {round(sum(fan_ins)/len(fan_ins), 2)}\n")
        for circ in circuits[-arch[-1]:]:
            f.write(f"{circ}\n")
    print("used:\n", learnt_arch, "\nout of:\n", arch)
    print(f"Max fan-in: {max(fan_ins)}\nAverage fan-in: {round(sum(fan_ins)/len(fan_ins), 2)}")
    return circuits[-arch[-1]:]

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
    for layer_i in range(i_1-1):
        next = jax.vmap(forward, in_axes=(None, 0))(xs, neurons[layer_i])
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
    for layer_i in range(i_1-1):
        next = jax.vmap(forward_disc, in_axes=(None, 0))(xs, neurons[layer_i])
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    return jax.vmap(forward_disc, in_axes=(None, 0))(xs, neurons[i_1-1])[:outs]

@jax.jit
def forward_conv(xs: jnp.ndarray, weights:jnp.ndarray, s: int, n: int) -> jnp.ndarray:
    """
    Applies a filter of width `w` and stride `s` to the input array `xs`.
    
    Parameters:
    weights - an array of shape (channels, new_n, new_n, old_channels, w, w), containing the filter weights
    xs - an array of shape (old_channels, n, n), the input data
    s - the stride of the filter
    
    Returns:
    An array of shape (new_n, new_n), the result of applying the filter.
    """
    w = weights.shape[2]
    old_channels = xs.shape[0]
    channels = jnp.arange(weights.shape[0])
    return jax.vmap(
        lambda c: jax.vmap(
            lambda i: jax.vmap(
                lambda j: f(jax.lax.dynamic_slice(xs, (0, i*s, j*s), (old_channels, w, w)), weights[c])
            )(jnp.arange(n.shape[0]))
        )(jnp.arange(n.shape[0]))
    )(channels)

@jax.jit
def forward_conv_disc(xs: jnp.ndarray, weights:jnp.ndarray, s: int, n: int) -> jnp.ndarray:
    """
    Applies a filter of width `w` and stride `s` to the input array `xs`.
    
    Parameters:
    weights - an array of shape (new_n, new_n, w, w), containing the filter weights
    xs - an array of shape (n, n), the input data
    s - the stride of the filter
    
    Returns:
    An array of shape (new_n, new_n), the result of applying the filter.
    """
    w = weights.shape[2]
    old_channels = xs.shape[0]
    channels = jnp.arange(weights.shape[0])
    return jax.vmap(
        lambda c: jax.vmap(
            lambda i: jax.vmap(
                lambda j: f_disc(jax.lax.dynamic_slice(xs, (0, i*s, j*s), (old_channels, w, w)), weights[c])
            )(jnp.arange(n.shape[0]))
        )(jnp.arange(n.shape[0]))
    )(channels)

@jax.jit
def feed_forward_conv(xs: jnp.ndarray, weights:jnp.ndarray) -> jnp.ndarray:
    """
    Applies all of the convolutional layers to the input
    
    Parameters:
    weights - the list of weights
    xs - an array of shape (n, n), the input data
    
    Returns:
    The result of applying the convolutional layers
    """
    for ws, (_,_,s,n) in zip(weights, convs):
        xs = forward_conv(xs, ws, s, jnp.zeros(n))
    return xs

@jax.jit
def feed_forward_conv_disc(xs: jnp.ndarray, weights:jnp.ndarray) -> jnp.ndarray:
    """
    Applies all of the convolutional layers to the input
    
    Parameters:
    weights - the list of weights
    xs - an array of shape (n, n), the input data
    
    Returns:
    The result of applying the convolutional layers
    """
    for ws, (_,_,s,n) in zip(weights, convs):
        xs = forward_conv_disc(xs, ws, s, jnp.zeros(n))
    return xs

def get_weights_conv(w: int, c: int, old_c: int, sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
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
    # layer lists, each with arch[i] elements
    # so this is a 2D list of floats
    # or a 1D list of jnp arrays
    mu = -jnp.log(old_c*w**2-1)/k
    return sigma * jax.random.normal(jax.random.key(key), shape=(c, old_c, w, w)) + mu #type: ignore

def initialise_conv(convs: List[Tuple[int, int, int, int]], sigma: jnp.ndarray, k: jnp.ndarray) -> Network:
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
    current_c = 1
    for w,_,c,_ in convs:
        weights = get_weights_conv(w, c, current_c, sigma, k)
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

def initialise(arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> List[jnp.ndarray]:
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
        layer = jnp.ones((arch[i1], i1, i_4))
        for i2 in range(arch[i1]):
            layer = layer.at[i2].set(get_weights(i1, arch, sigma, k))
        neurons.append(layer)
    return neurons

def get_shapes(arch: List[int]) -> Tuple[NetworkShape, int]:
    """
    returns a data structure which tells you the exact shape of the input wires for each NAND gate

    Parameters
    arch - a list representing the architecture
    
    Returns
    shapes - the data structure
    total - the total number of wires in the network
    """
    # gets the shape of the network based on the architecture
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
        fan_ins = jnp.concatenate((fan_ins, jax.vmap(lambda x:jnp.sum(jax.nn.sigmoid(x)))(layer)))
    l2s = jax.nn.relu(fan_ins-max_fan_in)
    return jnp.sum(jax.nn.softmax(l2s)*l2s)

@jax.jit
def cont_or(arr: jnp.ndarray) -> float:
    return 1-jnp.prod(1-arr)

@jax.jit
def get_l3(neurons: Network) -> float:
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
    sig_neurons = [jax.nn.sigmoid(layer[:,1:,:]) for layer in neurons]
    used_back = jnp.zeros(shape=(i_3, i_4))
    used_back = used_back.at[i_3-1, :outs].set(jnp.ones(shape=outs))
    # outputs are used by outputs
    used_back = used_back.at[:i_3-1].set(1-jnp.prod(1-sig_neurons[i_3-1], axis=0))
    for layer in range(i_3-2, -1, -1):
        temp = sig_neurons[layer][:arch[layer+1], :layer] * used_back[layer, :arch[layer+1]][:, jnp.newaxis, jnp.newaxis]
        used_back = used_back.at[:layer].set(1-(jnp.prod(1-temp, axis=0)*(1-used_back[:layer])))
    used_for = jnp.zeros(shape=(i_3, i_4))
    sig_neurons = [jax.nn.sigmoid(layer) for layer in neurons]
    input_con = jax.vmap(cont_or)(sig_neurons[0][:,0][:arch[1]])
    used_for = used_for.at[0, :arch[1]].set(input_con)
    for layer in range(1, i_3):
        #input_con - an array of length arch[layer+1], telling us how connected each neuron is directly to the inputs
        input_con = jax.vmap(cont_or)(sig_neurons[layer][:,0][:arch[layer+1]])
        temp = used_for[:layer][jnp.newaxis,:,:] * sig_neurons[layer][:,1:layer+1][:arch[layer+1], :layer]
        used_for = used_for.at[layer, :arch[layer+1]].set(1-(jnp.prod(1-temp, axis=(1,2))*(1-input_con)))
    return jax.nn.relu(jnp.sum(used_back*used_for) - max_gates)

epsilon = 1e-7
@jax.jit
def loss(neurons: Network, inputs: jnp.ndarray, output: jnp.ndarray, mask1: jnp.ndarray, mask2:jnp.ndarray) -> float:
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
        l2 = get_l2(neurons)
    else:
        l2 = 0
    if l3_coeff:
        l3 = get_l3(neurons)
    else:
        l3 = 0
    return l1 + l2_coeff*l2 + l3*l3_coeff

@jax.jit
def loss_conv(network: List[Network], inputs: jnp.ndarray, output: jnp.ndarray) -> float:
    """
    calculates loss

    Parameters
    network - [neurons, neurons_conv], where neurons are the dense layers, and neurons_conv are the convolutional
    inputs - all of the inputs (training xs)
    output - all of the outputs (training labels or ys)
    
    Returns
    loss
    """
    pred = jax.vmap(feed_forward_conv, in_axes=(0, None))(inputs, network[1])
    pred = pred.reshape(pred.shape[0], -1)
    if add_comp == 'y':
        pred = jnp.concatenate([pred, 1-pred], axis=1)
    pred = jax.vmap(feed_forward, in_axes=(0, None))(pred, network[0])
    pred = jnp.clip(pred, epsilon, 1-epsilon)
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    if l2_coeff:
        l2 = get_l2(network[0])
    else:
        l2 = 0
    return l1 + l2_coeff*l2

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
            [print(circ) for circ in (output_circuit(neurons, False, False))]
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
    pred = jax.vmap(feed_forward_conv_disc, in_axes=(0, None))(x_test, neurons_conv)
    pred = pred.reshape(pred.shape[0], -1)
    if add_comp == 'y':
        pred = jnp.concatenate([pred, 1-pred], axis=1)
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(pred, neurons)
    result = jax.vmap(image_class.evaluate)(pred, y_test)
    return jnp.sum(result)/result.size

i_1 = len(arch) - 1
# i_2 = max(arch[1:])
i_3 = i_1
i_4 = max(arch)
shapes, total = get_shapes(arch)
global_n = sum([arch[layer]*sum(arch[:layer]) for layer in range(1, len(arch))])/sum(arch)
print(global_n)
all_sigmas = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
all_ks = [1.0, 0.99, 0.98, 0.97, 0.955, 0.94, 0.92, 0.91, 0.9, 0.85, 0.75, 0.65, 0.5, 0.39, 0.32, 0.27, 0.23, 0.205, 0.18, 0.17, 0.155, 0.14, 0.13, 0.12, 0.11]

schedule = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0.05),
        optax.constant_schedule(0.01),
        optax.constant_schedule(0.003)
    ],
    boundaries=[2*batches, 4*batches]
)

solver = optax.adam(learning_rate=schedule)

if add_or_img == 'i':
    batch_schedule = [60000, 30000, 20000, 15000, 12000, 10000, 7500, 6000, 5000, 4000, 3750, 3000,
                      2500, 2400, 2000, 1875, 1500, 1250, 1200, 1000, 800, 750, 625, 600, 500]
    batch_i = 0

print("Learning:\n", output, "\nwith arch:", arch)
key = random.randint(0, 10000)
start_time = time.time()
if add_or_img == 'i':
    neurons = initialise(arch, all_sigmas[10], all_ks[10])
    neurons_conv = initialise_conv(convs, all_sigmas[4], all_ks[4])
else:
    neurons = initialise(arch, all_sigmas[sigma_i], all_ks[sigma_i])
if add_or_img == 'i':
    opt_state = solver.init([neurons, neurons_conv])
else:
    opt_state = solver.init(neurons)
init_time = time.time()
print("Took", init_time-start_time, "seconds to initialise.")
cont = True
if add_or_img == 'i':
    accuracy = acc_conv(neurons, neurons_conv)
    new_loss = loss_conv([neurons, neurons_conv], inputs, output)
    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),5)}")
    grad_conv = jax.jit(jax.grad(loss_conv, argnums=0))
else:
    accuracy = acc(neurons)
    new_loss = loss(neurons, inputs, output, jnp.array([]), jnp.array([]))
    print(f"Accuracy: {round(100*float(accuracy[0]),2)}%, Loss: {round(float(new_loss),5)}")
    grad = jax.jit(jax.grad(loss, argnums=0))
iters = 0
popped = 0
file_i = -1

restart = False

while cont:
    iters += 1
    if add_or_img == 'i':
        batches = batch_schedule[min(len(batch_schedule)-1, int(batch_i))]
        batch_i += 0.5
        batch_size = num_ins // batches
    for _ in range(max(10//batches, 1)):
        if batches > 1:
            key = random.randint(0, 10000)
            key = jax.random.PRNGKey(key)
            shuffled_indices = jax.random.permutation(key, inputs.shape[0])
            inputs = inputs[shuffled_indices]
            output = output[shuffled_indices]
        # batched_inputs = inputs.reshape(batches, batch_size, inputs.shape[1])
        # batched_output = output.reshape(batches, batch_size, output.shape[1])
        for batch in range(batches):
            if add_or_img == 'i':
                gradients = grad_conv([neurons, neurons_conv],
                                      inputs[batch*batch_size:(batch+1)*batch_size],
                                      output[batch*batch_size:(batch+1)*batch_size])
                update, opt_state = solver.update(gradients, opt_state, [neurons, neurons_conv])
                neurons, neurons_conv = optax.apply_updates([neurons, neurons_conv], update)
            elif weigh_even == 'y':
                gradients = grad(neurons,
                                 inputs[batch*batch_size:(batch+1)*batch_size],
                                 output[batch*batch_size:(batch+1)*batch_size],
                                 accuracy[1][batch*batch_size:(batch+1)*batch_size],
                                 accuracy[2][batch*batch_size:(batch+1)*batch_size])
                updates, opt_state = solver.update(gradients, opt_state, neurons)
                neurons = optax.apply_updates(neurons, updates)
                accuracy = acc(neurons)
            else:
                gradients = grad(neurons,
                                 inputs[batch*batch_size:(batch+1)*batch_size],
                                 output[batch*batch_size:(batch+1)*batch_size],
                                 jnp.array([]), jnp.array([]))
                updates, opt_state = solver.update(gradients, opt_state, neurons)
                neurons = optax.apply_updates(neurons, updates)
    if get_optional_input_non_blocking() == 4:
        batches = new_batches
        batch_size = num_ins // batches
    if add_or_img != 'i':
        if test(neurons) and (l2_coeff==0 or test_fan_in(neurons)) or get_optional_input_non_blocking() == 2:
            cont = False
    if cont:
        if add_or_img == 'i':
            new_loss = loss_conv([neurons, neurons_conv], inputs, output)
        else:
            if weigh_even == 'y':
                new_loss = loss(neurons, inputs, output, accuracy[1], accuracy[2])
            else:
                new_loss = loss(neurons, inputs, output, jnp.array([]), jnp.array([]))
            if get_optional_input_non_blocking() == 1:
                if add_or_img == 'i':
                    cont = False
                    print("Done training!")
                    print("Testing on testing data...")
                    accuracy = acc(neurons)
                    print(str(round(float(100*accuracy[0]),2))+"%% accuracy on the testing data")
                    image_class.save(neurons, convs, str(round(float(100*accuracy[0]),2))+'%', file_i)
                elif weigh_even == 'n':
                    print("Now weighing wrong more")
                    weigh_even = 'y'
                else:
                    restart = True
                    print("Restarting with new random weights")
                    neurons = initialise(arch, all_sigmas[sigma_i], all_ks[sigma_i])
                    new_loss = loss(neurons, inputs, output, jnp.array([]), jnp.array([]))
                    opt_state = solver.init(neurons)
            elif get_optional_input_non_blocking() == 3:
                if weigh_even == 'y':
                    weigh_even = 'n'
            if restart:
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
                print("Took", end_time-init_time, "seconds before restarting.")
                i_1 = len(arch) - 1
                i_2 = max(arch[1:])
                i_3 = i_1
                i_4 = max(arch)
                shapes, total = get_shapes(arch)
                global_n = sum([arch[layer]*sum(arch[:layer]) for layer in range(1, len(arch))])/sum(arch)
                neurons = initialise(arch, all_sigmas[sigma_i], all_ks[sigma_i])
                weigh_even = 'n'
                new_loss = loss(neurons, inputs, output, jnp.array([]), jnp.array([]))
                opt_stat = solver.init(neurons)
                init_time = time.time()
                print("New arch:", arch)
                accuracy = acc(neurons)
                print(f"Accuracy: {round(100*float(accuracy[0]),2)}%, Loss: {round(float(new_loss),5)}")
                print("Took", init_time-end_time, "seconds to initialise.")
        if iters == max(10//batches, 1):
            if add_or_img != 'i':
                accuracy = acc(neurons)
                print(f"Accuracy: {round(100*float(accuracy[0]),2)}%, Loss: {round(float(new_loss),5)}")
            else:
                accuracy = acc_conv(neurons, neurons_conv)
                print(f"Accuracy: {str(round(100*float(accuracy),2))}%, Loss: {round(float(new_loss),5)}")
                file_i = image_class.save(arch, neurons_conv, neurons, convs, str(round(float(100*accuracy),2))+'%', file_i)
            iters = 0
end_time = time.time()
print("Took", end_time-init_time, "seconds to train.")
if add_or_img != 'i':
    print("Learnt:\n", output, "\nwith arch:", arch)
    circuit = output_circuit(neurons)
    [print(circ) for circ in circuit]
