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

def get_optional_input_non_blocking():
    if os.name == 'nt':  # Windows
        if msvcrt.kbhit():
            user_input = msvcrt.getch().decode('utf-8').strip()
            if "s" in user_input:
                return True
    else:  # Unix-like systems
        input_ready, _, _ = select.select([sys.stdin], [], [], 0)  # Non-blocking select
        if input_ready:
            user_input = sys.stdin.readline().strip()
            if "s" in user_input:
                return True
    return False

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
    # for images, we're adding some copies of the inputs with convolution masks applied
    inputs, true_arch, convs = image_class.prep_in(inputs)
    x_test = image_class.prep_test(x_test, convs)
    ins = true_arch[0]
    new_ins = sum(true_arch)
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
print(true_arch)

new_ins = inputs.shape[1]

# I found using the same distribution for weights globally, and a sigma value of 1.5 works best, particularly for large networks
# which is what I'm setting up here for images. Still however adding the option for fine tuning for the other ones.
if add_or_img == 'i':
    global_weights = 'g'
    start_i = 10
    end_i = 11
    num = 1
else:
    global_weights = input("Global weights(g) or local(l)?\n")
    start_i = int(input("Input starting index (recommend 4 for local, 10 for global):\n"))
    end_i = int(input("Input ending index (max 24, recommend 5 for local, 11 for global):\n"))
    num = int(input("How many copies?\n"))

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
    wide = input("Add width(w) or depth(d)?\n")
    arch = [new_ins] + [width] * hidden + [outs]
else:
    starting_width = int(input("Starting Width:\n"))
    hidden = int(input("No. hidden layers:\n"))
    wide = input("Add width(w) or depth(d)?\n")
    diff = starting_width - outs
    layer_diff = diff/hidden
    arch = [new_ins] + [round(starting_width-i*layer_diff) for i in range(hidden)] + [outs]

# l2 pushes the weights towards +- infinity, I typically use 0 for this. I've found that the regular
# loss function without l2 works just fine.
l2_coeff = float(input("l2 coefficient:\n"))
l3_coeff = float(input("l3 coefficient:\n"))
max_fan_in = int(input("What should the max fan-in of the whole network be?:\n"))

# for adders and arbitrary combinational logic circuits, where we're aiming for 100% accuracy, if we're stuck
# in the high nineties at a local minima, I've added this to give a little nudge. It makes the losses of the
# incorrect samples weigh more.
weigh_even = 'n'

batches = int(input("How many batches?\n"))
batch_size = num_ins//batches
some_or_less = input("Some arrays(s) or less arrays(l)?\n")

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
def forward(weights: jnp.ndarray, xs: jnp.ndarray) -> float:
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
def forward_disc(weights: jnp.ndarray, xs: jnp.ndarray) -> int:
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
    print("used:\n", learnt_arch, "\nout of:\n", arch)
    print(f"Max fan-in: {max(fan_ins)}\nAverage fan-in: {round(sum(fan_ins)/len(fan_ins), 2)}")
    return circuits[-arch[-1]:]

# if some_or_less == 's':
#     with open("some_arrays.txt", 'r') as file:
#         exec(file.read())
# else:
#     with open("less_arrays.txt", 'r') as file:
#         exec(file.read())

if some_or_less == 's':
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
        xs = jnp.ones((i_3,i_4))
        xs = xs.at[0].set(jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1))
        for layer_i in range(i_1-1):
            xs = xs.at[layer_i+1, :arch[layer_i+1]].set(jax.vmap(forward, in_axes=(0, None))(neurons[layer_i], xs)[:arch[layer_i+1]])
        return jax.vmap(forward, in_axes=(0, None))(neurons[i_1-1], xs)[:outs]

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
        xs = jnp.ones((i_3,i_4))
        xs = xs.at[0].set(jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1))
        for layer_i in range(i_1-1):
            xs = xs.at[layer_i+1, :arch[layer_i+1]].set(jax.vmap(forward_disc, in_axes=(0, None))(neurons[layer_i], xs)[:arch[layer_i+1]])
        return jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:outs]

    def feed_forward_disc_print(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the discrete output of the network, whilst outputting useful debugging data

        Parameters
        inputs - the input data
        neurons - the network
        
        Returns
        the discrete output
        """
        xs = jnp.ones((i_3,i_4))
        xs = xs.at[0].set(jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1))
        jax.debug.print("inputs:{}", inputs)
        jax.debug.print("{}", xs)
        for layer_i in range(i_1-1):
            xs = xs.at[layer_i+1, :arch[layer_i+1]].set(jax.vmap(forward_disc, in_axes=(0, None))(neurons[layer_i], xs)[:arch[layer_i+1]])
            jax.debug.print("{}", xs)
        jax.debug.print("{}", jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:outs])
        return jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:outs]

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
        weights = jnp.ones((i_3,i_4)) * -jnp.inf
        # layer lists, each with arch[i] elements
        # so this is a 2D list of floats
        # or a 1D list of jnp arrays
        if global_weights == 'g':
            n = global_n
        else:
            n = sum(arch[:layer])
        mu = -jnp.log(n-1)/k
        for i in range(layer):
            inner_layer = sigma * jax.random.normal(jax.random.key(key), (arch[i])) + mu #type: ignore
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
            layer = jnp.ones((arch[i1], i_3, i_4))
            for i2 in range(arch[i1]):
                layer = layer.at[i2].set(get_weights(i1, arch, sigma, k))
            neurons.append(layer)
        return neurons
else:
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
            next = jax.vmap(forward, in_axes=(0, None))(neurons[layer_i], xs)
            next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
            xs = jnp.vstack([xs, next])
        return jax.vmap(forward, in_axes=(0, None))(neurons[i_1-1], xs)[:outs]

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
            next = jax.vmap(forward_disc, in_axes=(0, None))(neurons[layer_i], xs)
            next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
            xs = jnp.vstack([xs, next])
        return jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:outs]

    def feed_forward_disc_print(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the discrete output of the network, whilst outputting useful debugging data

        Parameters
        inputs - the input data
        neurons - the network
        
        Returns
        the discrete output
        """
        xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
        jax.debug.print("inputs:{}", inputs)
        jax.debug.print("{}", xs)
        for layer_i in range(i_1-1):
            next = jax.vmap(forward_disc, in_axes=(0, None))(neurons[layer_i], xs)
            next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
            xs = jnp.vstack([xs, next])
            jax.debug.print("{}", xs)
        jax.debug.print("{}", jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:outs])
        return jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:outs]

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
def get_l3(neurons: Network) -> float:
    """
    calculates l3, which is minimised for any maximum fan-in under or equal to "max_fan_in"

    Parameters
    neurons - the network
    
    Returns
    l3
    """
    l3s = jnp.array([])
    for layer in neurons:
        l3s = jnp.concatenate((l3s, jax.vmap(lambda x:jnp.sum(jax.nn.sigmoid(x)))(layer)))
    raw = jnp.sum(jax.nn.softmax(l3s)*l3s)
    return jax.nn.relu(raw-max_fan_in+1)

@jax.jit
def get_l2(neurons: Network) -> float:
    """
    calculates l2, which is minimised for extreme weights (close to +-inf)

    Parameters
    neurons - the network
    
    Returns
    s/total - l2
    """
    s = 0
    for layer in neurons:
        s += jnp.sum(1-jax.nn.sigmoid(jnp.absolute(layer)))
    return s/total

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
    return l1 + l2_coeff*l2 + l3_coeff*l3

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
    if add_or_img == 'i':
        pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(x_test, neurons)
        result = jax.vmap(image_class.evaluate)(pred, y_test)
        return jnp.sum(result)/result.size, jnp.zeros(0), jnp.zeros(0)
    else:
        pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neurons)
        pred = (pred == output)
        pred = jnp.sum(pred, axis=1)
        if weigh_even == 'y':
            trues = jnp.where(pred == outs)
            falses = jnp.where(pred < outs)
            return jnp.sum(pred)/((2**(ins))*(outs)), trues[0], falses[0]
        return jnp.sum(pred)/((2**(ins))*(outs)), jnp.zeros(0), jnp.zeros(0)

i_1 = len(arch) - 1
# i_2 = max(arch[1:])
i_3 = i_1
i_4 = max(arch)
shapes, total = get_shapes(arch)
global_n = sum([arch[layer]*sum(arch[:layer]) for layer in range(1, len(arch))])/sum(arch)
print(global_n)
all_sigmas = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
all_ks = [1.0, 0.99, 0.98, 0.97, 0.955, 0.94, 0.92, 0.91, 0.9, 0.85, 0.75, 0.65, 0.5, 0.39, 0.32, 0.27, 0.23, 0.205, 0.18, 0.17, 0.155, 0.14, 0.13, 0.12, 0.11]
sigmas = jnp.array(all_sigmas[start_i:end_i] * num)
ks = jnp.array(all_ks[start_i:end_i] * num)
n = sigmas.shape[0]

print("Learning:\n", output, "\nwith arch:", arch)
key = random.randint(0, 10000)
start_time = time.time()
neuronss = [initialise(arch, sigmas[i], ks[i]) for i in range(n)]
solver = optax.adam(learning_rate=0.003)
opt_states = [solver.init(neurons) for neurons in neuronss]
restart_mask = [False] * n
init_time = time.time()
print("Took", init_time-start_time, "seconds to initialise.")
# print(opt_states[0])
cont = True
print("Accuracies:")
accuracies = [acc(neurons) for neurons in neuronss]
print(accuracies)
print([str(round(100*float(accuracy[0]),2))+'%' for accuracy in accuracies])
if weigh_even == 'y':
    old_losses = [loss(neuronss[index], inputs, output, accuracies[index][1], accuracies[index][2]) for index in range(n)]
else:
    old_losses = [loss(neuronss[index], inputs, output, jnp.array([]), jnp.array([])) for index in range(n)]
print("Losses:")
print([round(float(old_loss),5) for old_loss in old_losses])
grad = jax.jit(jax.grad(loss))
iters = 0
popped = 0
file_i = -1

while cont:
    iters += 1
    for _ in range(max(10//batches, 1)):
        for i in range(n-popped):
            for batch in range(batches):
                if weigh_even == 'y':
                    updates, opt_states[i] = solver.update(grad(neuronss[i], inputs[batch*batch_size:(batch+1)*batch_size], output[batch*batch_size:(batch+1)*batch_size], accuracies[i][1][batch*batch_size:(batch+1)*batch_size], accuracies[i][2][batch*batch_size:(batch+1)*batch_size]), opt_states[i], neuronss[i])
                    neuronss[i] = optax.apply_updates(neuronss[i], updates)
                    
                else:
                    updates, opt_states[i] = solver.update(grad(neuronss[i], inputs[batch*batch_size:(batch+1)*batch_size], output[batch*batch_size:(batch+1)*batch_size], jnp.array([]), jnp.array([])), opt_states[i], neuronss[i])
                    neuronss[i] = optax.apply_updates(neuronss[i], updates)
                accuracies = [acc(neurons) for neurons in neuronss]
    for index, neurons in enumerate(neuronss):
        if test(neurons) and (l3_coeff==0 or test_fan_in(neurons)):
            print(index)
            # jax.debug.print("neurons:{}", neurons)
            # print(jax.vmap(feed_forward_disc_print, in_axes=(0, None))(inputs, neuronss[index]))
            final_index = index
            cont = False
    if cont:
        if weigh_even == 'y':
            new_losses = [loss(neuronss[index], inputs, output, accuracies[index][1], accuracies[index][2]) for index in range(n-popped)]
        else:
            new_losses = [loss(neuronss[index], inputs, output, jnp.array([]), jnp.array([])) for index in range(n-popped)]
        for i in range(n-popped):
            if old_losses[i] == new_losses[i] or get_optional_input_non_blocking():
                old_losses[i] = new_losses[i]
                if add_or_img == 'i':
                    cont = False
                    print("Done training!")
                    print("Testing on testing data...")
                    accuracy = acc(neuronss[i])
                    print(str(round(float(100*accuracy[0]),2))+"%% accuracy on the testing data")
                    image_class.save(neuronss[i], convs, str(round(float(100*accuracy[0]),2))+'%', file_i)
                elif weigh_even == 'n':
                    print("Now weighing wrong more")
                    weigh_even = 'y'
                else:
                    restart_mask[i] = True
                    print("Restarting sigma value", sigmas[i], "with new random weights, stuck in local minima", new_losses[i], old_losses[i])
                    # print(output_circuit(neuronss[i]))
                    # print(jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neuronss[i]))
                    neuronss[i] = initialise(arch, sigmas[i], ks[i])
                    if weigh_even == 'y':
                        old_losses[i] = loss(neuronss[i], inputs, output, accuracies[i][1], accuracies[i][2])
                    else:
                        old_losses[i] = loss(neuronss[i], inputs, output, jnp.array([]), jnp.array([]))
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
            sigmas = jnp.array(all_sigmas[start_i:end_i] * num)
            ks = jnp.array(all_ks[start_i:end_i] * num)
            neuronss = [initialise(arch, sigmas[i], ks[i]) for i in range(n)]
            weigh_even = 'n'
            new_losses = [loss(neuronss[index], inputs, output, jnp.array([]), jnp.array([])) for index in range(n)]
            old_losses = new_losses.copy()
            opt_states = [solver.init(neurons) for neurons in neuronss]
            init_time = time.time()
            print("New arch:", arch)
            print("Accuracies:")
            accuracies = [acc(neurons) for neurons in neuronss]
            print([str(round(100*float(accuracy[0]),2))+'%' for accuracy in accuracies])
            print("Losses:")
            print([round(float(old_loss),5) for old_loss in old_losses])
            print("Took", init_time-end_time, "seconds to initialise.")
        if iters == max(10//batches, 1):
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
                if accuracies[to_pop][0] == min([a[0] for a in accuracies]) or new_losses[to_pop] >= 2*min(new_losses):
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
            accuracies = [acc(neurons) for neurons in neuronss]
            print("Accuracies:")
            print([str(round(100*float(accuracy[0]),2))+'%' for accuracy in accuracies])
            print("Losses:")
            print([round(float(old_loss),5) for old_loss in old_losses])
            i = old_losses.index(min(old_losses))
            if add_or_img == 'i':
                file_i = image_class.save(extra_layers, arch, some_or_less, neuronss[i], convs, str(round(float(100*accuracies[i][0]),2))+'%', file_i)
            iters = 0
end_time = time.time()
print("Took", end_time-init_time, "seconds to train.")
if add_or_img != 'i':
    print("Learnt:\n", output, "\nwith arch:", arch)
    circuit = output_circuit(neuronss[final_index])
    [print(circ) for circ in circuit]
    print(sigmas[final_index])
