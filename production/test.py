import jax
import jax.numpy as jnp
import optax
import random
import itertools
from typing import List, Tuple, Set, Dict, Callable
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

def run_test(variables: Dict[str, any]):
    with open("set-up.yaml", "r") as f:
        config = yaml.safe_load(f)

    for k,v in variables.items():
        config[k] = v

    print(variables)
    jax.config.update("jax_traceback_filtering", config["traceback"])

    # defining some types
    Network = Tuple[jnp.ndarray, ...]
    Shape = Tuple[int, ...]

    print(jax.devices())

    add_img_or_custom = config["add_img_or_custom"]

    if add_img_or_custom == 'c':
        import utils.custom_util as custom_util
        inputs, output, ins, outs, num_ins = custom_util.set_up_custom(config)
    elif add_img_or_custom == 'a':
        import utils.adders_util as adders_util
        inputs, output, ins, outs, num_ins = adders_util.set_up_adders(config)
    else:
        import utils.image_util as image_util
        inputs, x_test, output, y_test, num_ins = image_util.set_up_img(config)
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
            new_ins = convs[-1][2] * convs[-1][3]**2 + 2*config["size"]**2
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

    i_0 = len(true_arch) - 1
    # i_1 = max(true_arch[1:])
    # i_2 = i_0
    i_3 = max(true_arch)

    weights_shape = []
    for i in range(1, len(true_arch)):
        if i <= 3 or i == len(true_arch)-1:
            weights_shape.append((sum(true_arch[:i]), true_arch[i]))
        else:
            weights_shape.append(
                (true_arch[0]+true_arch[i-2]+true_arch[i-1], true_arch[i]))
    global_n = (sum(ns[0]*ns[1] for ns in weights_shape) 
                / sum(ns[1] for ns in weights_shape))

    num_wires = sum(ns[0]*ns[1] for ns in weights_shape)

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
    max_gates_used_penalty_coeff = float(max_gates_used_penalty_coeff
                                        / (sum(arch)-sum(max_gates)))
    continuous_penalty_coeff = config["continuous_penalty_coeff"]
    min_gates_used_penalty_coeff = config["min_gates_used_penalty_coeff"]
    if min_gates_used_penalty_coeff == 0:
        min_gates = jnp.array([0]*len(arch))
    else:
        min_gates = jnp.array(config["min_gates"])
        min_gates_used_penalty_coeff = float(min_gates_used_penalty_coeff
                                            / (sum(min_gates)))
    mean_fan_in_penalty_coeff = config["mean_fan_in_penalty_coeff"]
    if mean_fan_in_penalty_coeff == 0:
        mean_fan_in = 0
    else:
        mean_fan_in = config["mean_fan_in"]

    dps = config["decimal_places"]

    step = jax.jit(lambda x: jnp.where(x>0, 1, 0))

    def sample(seed, p):
        key = jax.random.fold_in(jax.random.PRNGKey(0), seed)
        return jax.random.bernoulli(key, p)

    @jax.jit
    def bern(x):
        x = jnp.asarray(x, dtype=jnp.float32)
        seeds = jax.lax.bitcast_convert_type(x, jnp.uint32).ravel()
        probs = jax.nn.sigmoid(x).ravel()
        samples_flat = jax.vmap(sample)(seeds, probs)
        return samples_flat.reshape(x.shape)
    
    temp = jax.jit(lambda x: jax.nn.sigmoid(x/temperature))
    weight_activation_dict = {"cont": jax.nn.sigmoid,
                            "disc": step,
                            "rand": bern,
                            "temp": temp}

    @partial(jax.jit, static_argnames="weight_activation")
    def and_helper(
        x: jnp.ndarray,
        w: jnp.ndarray,
        weight_activation: str="cont") -> float:
        """
        Helper function for forward, calculates the effective input a neuron
        receives from a specific previous layer (which is effectively a logical
        AND)

        Parameters
        x - could be inputs, could be outputs from a previous NAND gate,
        importantly it's a jnp array all from the same layer
        w - the weights of those wires connecting x to the NAND gate
        weight_activation - a string which is "cont" or "disc", which determines
        if we use a sigmoid or a step function
        
        Returns
        the effective input from that layer for the NAND gate
        """
        return jnp.prod(
            1 + jnp.multiply(x, weight_activation_dict[weight_activation](w))
            - weight_activation_dict[weight_activation](w))

    @partial(jax.jit, static_argnames="weight_activation")
    def forward(
        xs: jnp.ndarray, ws: jnp.ndarray, weight_activation: str="cont") -> float:
        """
        The forward pass for a neuron

        Parameters
        xs - a 2d jnp array of all the values on those wires
        ws - a 2d jnp array of all the wires going into it
        weight_activation - a string which is "cont" or "disc", which determines
        if we use a sigmoid or a step function
        
        Returns
        the continuous effective output for that NAND gate
        """
        return 1 - jnp.prod(jax.vmap(
            and_helper, in_axes=(0, 0, None))(xs, ws, weight_activation))

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

    @partial(jax.jit, static_argnames=("weight_activation", "use_surr"))
    def feed_forward(
        inputs: jnp.ndarray,
        weights: jnp.ndarray,
        weight_activation: str="cont",
        use_surr: bool=False,
        surr_arr: List[jnp.ndarray]=[]) -> jnp.ndarray:
        """
        Calculates the output of the network

        Parameters
        inputs - the input data
        weights - the network
        weight_activation - a string which is "cont" or "disc", which determines
        if we use a sigmoid or a step function
        use_surr - boolean value for if we're adding surrogate bits
        surr_arr - how to calculate the surrogate bits for if we're using them
        
        Returns
        the output of the network
        """
        xs = jnp.array([jnp.pad(
            inputs,(0, i_3-len(inputs)), mode="constant", constant_values=1)])
        for layer_i in range(min(i_0-1, 3)):
            next = jax.vmap(forward, in_axes=(None, 0, None))(
                xs, weights[layer_i], weight_activation)
            if use_surr and layer_i < len(surr_arr):
                next = jnp.concatenate([calc_surr(xs, layer_i, surr_arr), next])
            next = jnp.array([jnp.pad(
                next,(0, i_3-len(next)), mode="constant", constant_values=1)])
            xs = jnp.vstack([xs, next])
        for layer_i in range(3, i_0-1):
            next = jax.vmap(forward, in_axes=(None, 0, None))(
                xs[jnp.array([0,-2,-1])], weights[layer_i], weight_activation)
            if use_surr and layer_i < len(surr_arr):
                next = jnp.concatenate([calc_surr(xs, layer_i, surr_arr), next])
            next = jnp.array([jnp.pad(
                next,(0, i_3-len(next)), mode="constant", constant_values=1)])
            xs = jnp.vstack([xs, next])
        return jax.vmap(
            forward, in_axes=(None, 0, None))(
                xs, weights[i_0-1], weight_activation)[:outs]

    if add_img_or_custom == 'i':
        @partial(jax.jit, static_argnames=('n', "weight_activation"))
        def forward_conv(
            xs: jnp.ndarray,
            weights:jnp.ndarray,
            s: int,
            n: int,
            weight_activation: str="cont") -> float:
            """
            Applies a filter of width `w` and stride `s` to the input array `xs`.
            
            Parameters:
            xs - an array of shape (old_channels, old_n, old_n), the input data
            weights - an array of shape (channels, old_channels, w, w), containing
            the filter weights
            s - the stride of the filter
            n - the new height and width of the picture
            weight_activation - a string which is "cont" or "disc", which
            determines if we use a sigmoid or a step function

            Returns:
            An array of shape (channels, n, n), the result of applying the filter.
            """
            w = weights.shape[2]
            old_channels = xs.shape[0]
            channels = jnp.arange(weights.shape[0])
            return jax.vmap(
                lambda c: jax.vmap(
                    lambda i: jax.vmap(
                        lambda j: 1-and_helper(
                            jax.lax.dynamic_slice(xs,
                            (0, i*s, j*s),
                            (old_channels, w, w)),
                            weights[c], weight_activation)
                    )(jnp.arange(n))
                )(jnp.arange(n))
            )(channels)

        # depends on convs
        @partial(jax.jit, static_argnames="weight_activation")
        def feed_forward_conv(
            xs: jnp.ndarray,
            weights:jnp.ndarray,
            imgs_list: List[jnp.ndarray],
            weight_activation: str="cont") -> jnp.ndarray:
            """
            Applies all of the convolutional layers to the input
            
            Parameters:
            xs - an array of shape (n, n), the input data
            weights - the list of weights
            imgs_list - a list of the scaled down images
            weight_activation - a string which is "cont" or "disc", which
            determines if we use a sigmoid or a step function
            
            Returns:
            The result of applying the convolutional layers, ready to be passed
            into the dense layers
            """
            for i, (ws, (_,_,s,n)) in enumerate(zip(weights, convs)):
                temp = forward_conv(xs, ws, s, n, weight_activation)
                xs = jnp.concatenate(
                    [imgs_list[i], 1-imgs_list[i], temp, 1-temp], axis=0)
            return xs

        convs = tuple(tuple(conv) for conv in convs)

    def get_used(used: List[int], arch: Tuple[int, ...], verbose: bool) -> List[int]:
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

    def clean_connected(connetecteds: Dict[int, List[int]],
                        used_list: List[int],
                        arch: Tuple[int, ...]) -> List[List[jnp.ndarray]]:
        """
        Converts our connected dictionary, along with a list of the nodes that we
        used, into a data structure representing the learnt NAND network

        Parameters
        connecteds - A dictionary mapping the global index of the NAND gates to the
        list of nodes that connect to it
        used_list - a list of the global indices of nodes used in the learnt NAND
        network
        arch - a list of how many nodes are in each layer of the NAND network

        Returns
        a data structure representing the learnt NAND network, that can be used
        for simulating inference (see calc_surr for example).
        """
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
                connections = [
                    node_to_true_index[con] for con in connetecteds[node]]
                layer.append(connections)
        net.append(layer)
        return net[1:]

    def output_circuit(weights: Network, verbose=True, super_verbose=False,
                    weight_activation: str="disc") -> List[str]:
        """
        Outputs the learnt circuit, and also prints some useful data about the
        network
        
        Parameters
        weights - the internal representation of the circuit as learnt
        verbose - a flag for printing extra info
        super_verbose - a flag for even more debug info
        
        Returns
        circuits[-arch[-1]:] - a list of the circuit learnt for each output neuron
        """
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
                circuits, connecteds = adders_util.update_circuits(
                    add_adder_help, circuits, with_nots, connecteds)
        else:
            circuits = [chr(ord('A')+i) for i in range(arch[0])]
        gates = [[[] for _ in range(arch[0])]]
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
        for layer_i in range(i_0):
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
                                if node[:2] == "¬¬" and node[2] != '(':
                                    node = node[2:]
                        else:
                            node = ('¬(' +
                                    '.'.join([element[1] for element in connected])
                                    + ')')
                        if node in c2i.keys():
                            if layer_i == i_0-1:
                        
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
                            gates[-1].append(
                                [index2gate[element[0]] for element in connected])
                            index2gate[added] = (gate_i1, gate_i2)
                            gate_i2 += 1
                            if layer_i == i_0-1:
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
                        if (weight_activation_dict[weight_activation](
                            weights[layer_i][neuron_i,inner_layer_i,weight_i])
                            and indices[i] not in empties):
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
                            if node[:2] == "¬¬" and node[2] != '(':
                                node = node[2:]
                    else:
                        node = '¬(' + '.'.join(
                            [element[1] for element in sorted_connected]) + ')'
                    if node in c2i.keys():
                        if layer_i == i_0-1:
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
                        gates[-1].append([index2gate[element[0]]
                                        for element in sorted_connected])
                        index2gate[added] = (gate_i1, gate_i2)
                        gate_i2 += 1
                        if layer_i == i_0-1:
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
        print(true_weights)
        learnt_arch = get_used(used_list, true_arch, verbose)
        fan_ins = []
        for node_index in used_list:
            if node_index >= learnt_arch[0]:
                fan_ins.append(len(connecteds[node_index]))
        print("used:\n", learnt_arch, "\nout of:\n", true_arch)
        print(f"Max fan-in: {max(fan_ins)}\nAverage fan-in: {round(sum(fan_ins)/len(fan_ins), 2)}")
        with open(config["output_file"], "a") as f:
            f.write(f"used:\n{learnt_arch}\nout of:\n{true_arch}\n")
            f.write(f"Max fan-in: {max(fan_ins)}\nAverage fan-in: {round(sum(fan_ins)/len(fan_ins), 2)}\n")
            f.write(f"Used {weight_activation}\n")
            for circ in circuits[-true_arch[-1]:]:
                f.write(f"{circ}\n")
        return circuits[-true_arch[-1]:]

    def beta_sampler(shape: Shape, n: int, sigma: float, k: float=None
                    ) -> jnp.ndarray:
        """
        returns a set of numbers with the appropriate distribution. sigma must be
        at most sqrt(n-1)/n.
        This distribution ensures that the expected value of the sigmoid is 1/n
        
        Parameters
        shape - the shape of the array we want to return
        n - the number of wires into the NAND gate
        sigma - the standard deviation of the *beta* distribution

        Returns
        the weights
        """
        sigma = sigma * jnp.sqrt(n-1)/n
        key = random.randint(0, 10000)
        alpha = ((n-1)/(n**2*sigma**2)-1)/n
        beta = alpha * (n - 1)
        samples = jax.random.beta(
            jax.random.key(key), a=alpha, b=beta, shape=shape)
        samples = jnp.clip(samples, epsilon, 1-epsilon)
        return jnp.log(samples / (1 - samples))

    def normal_sampler1(shape: Shape, n: int, sigma: float, k: float=None
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
        mu = (-sigma)*jsp_special.ndtri((n-1) / n)
        return sigma * jax.random.normal(jax.random.key(key), shape=shape) + mu

    def normal_sampler2(shape: Shape, n: int, sigma: float, k: float
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
            distribution: Callable[[Shape, int, float, float], jnp.ndarray],
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
        a 2d jnp array of the weights, which represents the wires going into a
        certain neuron
        """
        n = old_c*w**2
        return distribution(shape=(c, old_c, w, w), n=n, sigma=sigma, k=k)

    def initialise_conv(
            convs: List[Tuple[int, int, int, int]],
            distribution: Callable[[Shape, int, float, float], jnp.ndarray],
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
        weights = []
        current_c = 2
        for w,_,c,_ in convs:
            layer_weights = get_weights_conv(
                w, c//2-1, current_c, distribution, sigma, k)
            weights.append(layer_weights)
            current_c = c
        return tuple(weights)

    def get_weights(
            layer: int,
            arch: Tuple[int, ...],
            distribution: Callable[[Shape, int, float, float], jnp.ndarray],
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
        if layer == 1 or layer == 2 or layer == len(arch)-1:
            weights = jnp.ones((layer, i_3)) * -jnp.inf
        else:
            weights = jnp.ones((3,i_3)) * -jnp.inf
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
                    (0, i_3-arch[i]),
                    mode="constant", constant_values=-jnp.inf))
        else:
            inner_layer = distribution(shape=(arch[0],), n=n, sigma=sigma, k=k)
            weights = weights.at[0].set(jnp.pad(
                inner_layer,
                (0, i_3-arch[0]),
                mode="constant", constant_values=-jnp.inf))
            for i in range(1,3):
                inner_layer = distribution(
                    shape=(arch[layer-3+i],), n=n, sigma=sigma, k=k)
                weights = weights.at[i].set(jnp.pad(
                    inner_layer,
                    (0, i_3-arch[layer-3+i]),
                    mode="constant", constant_values=-jnp.inf))
        return weights

    def initialise(
            arch: Tuple[int, ...],
            true_arch: Tuple[int, ...],
            distribution: Callable[[Shape, int, float, float], jnp.ndarray],
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
        weights = []
        for i1 in range(1, len(arch)):
            if i1 == 1 or i1 == 2 or i1 == len(arch) - 1:
                layer = jnp.ones((arch[i1], i1, i_3)) * (-jnp.inf)
            else:
                layer = jnp.ones((arch[i1], 3, i_3)) * (-jnp.inf)
            for i2 in range(arch[i1]):
                layer = layer.at[i2].set(get_weights(
                    i1, true_arch, distribution, sigma, k))
            weights.append(layer)
        return tuple(weights)

    @jax.jit
    def max_fan_in_penalty_disc(weights: Network, max_fan_in: int) -> float:
        """
        calculates a penalty, which is minimised for any maximum fan-in under or
        equal to "max_fan_in"
        this doesn't account for duplicate gates, and this is the discrete version.

        Parameters
        weights - the network
        max_fan_in - the desired maximum fan-in
        
        Returns
        the penalty
        """
        fan_ins = jnp.array([])
        for layer in weights:
            fan_ins = jnp.concatenate((fan_ins, jax.vmap(
                lambda x:jnp.sum(jnp.where(x>0, 1, 0)))(layer)))
        temp = jax.nn.relu(fan_ins-max_fan_in)
        return jnp.max(temp)

    @jax.jit
    def max_fan_in_penalty_rand(weights: Network, max_fan_in: int) -> float:
        """
        calculates a penalty, which is minimised for any maximum fan-in under or
        equal to "max_fan_in"
        this doesn't account for duplicate gates, and this is the discrete version.

        Parameters
        weights - the network
        max_fan_in - the desired maximum fan-in
        
        Returns
        the penalty
        """
        fan_ins = jnp.array([])
        for layer in weights:
            fan_ins = jnp.concatenate((fan_ins, jax.vmap(
                lambda x:jnp.sum(bern(x)))(layer)))
        temp = jax.nn.relu(fan_ins-max_fan_in)
        return jnp.max(temp)

    @jax.jit
    def max_fan_in_penalty(weights: Network, max_fan_in: int, temperature: float
                        ) -> float:
        """
        calculates a penalty, which is minimised for any maximum fan-in under or
        equal to "max_fan_in". This doesn't account for duplicate gates

        Parameters
        weights - the network
        max_fan_in - the desired maximum fan-in
        temperature - lower makes it closer to discrete
        
        Returns
        the penalty (a float, which will be multiplied by a coefficient, and added
        to the loss)
        """
        fan_ins = jnp.array([])
        for layer in weights:
            fan_ins = jnp.concatenate((fan_ins, jax.vmap(
                lambda x:jnp.sum(jax.nn.sigmoid(x/temperature)))(layer)))
        temp = jax.nn.relu(fan_ins-max_fan_in)
        return jnp.sum(jax.nn.softmax(temp)*temp)

    @jax.jit
    def mean_fan_in_penalty(
        weights: Network,
        mean_fan_in: float,
        temperature: float) -> float:
        """
        calculates a penalty, which is minimised for any mean fan-in under or equal
        to "mean_fan_in". This doesn't account for duplicate gates

        Parameters
        weights - the network
        mean_fan_in - the desired mean fan-in
        temperature - lower makes it closer to discrete
        
        Returns
        the penalty (a float, whihc will be multiplied by a coefficient, and added
        to the loss)
        """
        fan_ins = jnp.array([])
        for layer in weights:
            fan_ins = jnp.concatenate((fan_ins, jax.vmap(
                lambda x:jnp.sum(jax.nn.sigmoid(x/temperature)))(layer)))
            fan_ins = jnp.concatenate((fan_ins, jnp.zeros(i_3-layer.shape[0])))
        usage = get_used_array(weights, "temp")[1:]
        temp = jnp.sum(fan_ins * usage.reshape(-1))/jnp.sum(usage)
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
    def get_used_array(weights: Network, weight_activation: str) -> float:
        """
        returns an array, used, representing the network, where if
        used[layer][i] is close to 1, the neuron is used.
        used_back tells us if the NAND gates will be used in the outputs
        used_for tells us if the NAND gates are connected to the inputs
        and so their product tells us if the NAND gates are being used
        the doesn't account for duplicate gates

        Parameters
        weights - the network
        weight_activation - sigmoid with temperature or step function
        
        Returns
        the array
        """
        prob_weights = [
            weight_activation_dict[weight_activation](layer) for layer in weights]
        used_back = jnp.zeros(shape=(len(arch), i_3))
        used_back = used_back.at[len(arch)-1, :outs].set(jnp.ones(shape=outs))
        for layer in range(len(arch)-1, 0, -1):
            temp = (prob_weights[layer-1]
                    * used_back[layer, :arch[layer]][:, jnp.newaxis, jnp.newaxis])
            # this is a 2D matrix, the LHS of the * is how much each neuron to the
            # left of this neuron is used by this neuron. The RHS of the * is a
            # vector, which is how much this neuron is used by the output.
            temp = cont_or_arr(temp, axis=0)
            used_back = used_back.at[input_layers(layer)].set(
                cont_or(used_back[input_layers(layer)], temp))
        used_for = jnp.zeros(shape=(len(arch), i_3))
        used_for = used_for.at[0, :new_ins].set(jnp.ones(shape=new_ins))
        for layer in range(1, len(arch)):
            temp = (prob_weights[layer-1][:arch[layer]]
                    * used_for[input_layers(layer)][jnp.newaxis,:,:])
            temp = cont_or_arr(temp, axis=(1,2))
            used_for = used_for.at[layer, :arch[layer]].set(
                cont_or(used_for[layer, :arch[layer]], temp))
        return used_back*used_for

    @jax.jit
    def max_gates_used_penalty(weights: Network, max_gates: jnp.ndarray) -> float:
        """
        calculates a penalty, which is maximised for any gate usage less than or
        equal to "max_gates" this doesn't account for duplicate gates.

        Parameters
        weights - the network
        max_gates - an array specifying the max number of nodes in each layer.
        weight_activation - sigmoid with a temperature or step function
        
        Returns
        the penalty (a float, which will be multiplied by a coefficient, and added
        to the loss)
        """
        used = get_used_array(weights, "temp")
        return jnp.sum(jax.nn.relu(jnp.sum(used, axis=1)-max_gates))

    @jax.jit
    def min_gates_used_penalty(weights: Network, min_gates: jnp.ndarray) -> float:
        """
        calculates a penalty, which is maximised for any gate usage greater than or
        equal to "min_gates". This doesn't account for duplicate gates.

        Parameters
        weights - the network
        min_gates - an array specifying the min number of nodes in each layer.
        weight_activation - sigmoid with a temperature or step function
        
        Returns
        the penalty (a float, which will be multiplied by a coefficient, and added
        to the loss)
        """
        used = get_used_array(weights, "temp")
        return jnp.sum(jax.nn.relu(min_gates-jnp.sum(used, axis=1)))

    @partial(jax.jit, static_argnames="weight_activation")
    def gate_usage_by_layer(weights: Network, weight_activation: str) -> float:
        # gives us the gate usage by layer
        return jnp.sum(get_used_array(weights, weight_activation), axis=1)

    @jax.jit
    def continuous_penalty(weights: Network, num_wires: int) -> float:
        """
        calculates a penalty which is minimised when the weights have a high
        magnitude. Adding this to the loss can lead to networks where a low loss is
        more strongly correlated with a high accuracy

        Parameters
        weights - the network
        total - the number of weights in the network
        
        Returns
        the penalty (a float, which will be multiplied by a coefficient, and added
        to the loss)
        """
        s = sum([jnp.sum(
            1-jax.nn.sigmoid(jnp.absolute(layer))) for layer in weights])
        return s/num_wires

    epsilon = 1e-7
    @partial(jax.jit, static_argnames="use_surr")
    def bce_loss(
        weights: Network,
        inputs: jnp.ndarray,
        output: jnp.ndarray,
        mask1: jnp.ndarray=None,
        mask2:jnp.ndarray=None,
        use_surr: bool=False,
        surr_arr: List[jnp.ndarray]=[]) -> float:
        """
        calculates the binary cross entropy loss

        Parameters
        weights - the network
        inputs - all of the inputs (training xs)
        output - all of the outputs (training labels or ys)
        mask1 - a mask for samples we got right
        mask2 - a mask for the samples we got wrong
        
        Returns
        bce loss
        """
        pred = jax.vmap(feed_forward, in_axes=(0, None, None, None, None))(
            inputs, weights, "cont", use_surr, surr_arr)
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
        
    @partial(jax.jit, static_argnames="use_surr")
    def loss(
        weights: Network,
        inputs: jnp.ndarray,
        output: jnp.ndarray,
        mask1: jnp.ndarray=None,
        mask2:jnp.ndarray=None,
        use_surr: bool=False,
        surr_arr: List[jnp.ndarray]=[],
        max_fan_in: int=None,
        temperature: float=None,
        mean_fan_in: float=None,
        max_gates: jnp.ndarray=None,
        min_gates: jnp.ndarray=None,
        num_wires: int=None) -> float:
        """
        calculates the loss

        Parameters
        weights - the network
        inputs - all of the inputs (training xs)
        output - all of the outputs (training labels or ys)
        mask1 - a mask for samples we got right
        mask2 - a mask for the samples we got wrong
        
        Returns
        loss
        """
        l = bce_loss(weights, inputs, output, mask1, mask2, use_surr, surr_arr)
        if max_fan_in_penalty_coeff:
            l += (max_fan_in_penalty_coeff
                * max_fan_in_penalty(weights, max_fan_in, temperature))
        if mean_fan_in_penalty_coeff:
            l += (mean_fan_in_penalty_coeff
                * mean_fan_in_penalty(
                    weights, mean_fan_in, temperature))
        if max_gates_used_penalty_coeff:
            l += (max_gates_used_penalty_coeff
                * max_gates_used_penalty(weights, max_gates))
        if min_gates_used_penalty_coeff:
            l += (min_gates_used_penalty_coeff
                * min_gates_used_penalty(weights, min_gates))
        if continuous_penalty_coeff:
            l += continuous_penalty_coeff * continuous_penalty(weights, num_wires)
        return l

    grad = jax.jit(jax.grad(loss), static_argnames="use_surr")

    if add_img_or_custom=='i':

        @jax.jit
        def loss_conv(
            network: List[Network],
            inputs: jnp.ndarray,
            output: jnp.ndarray,
            scaled: List[jnp.ndarray]=None,
            max_fan_in: int=None,
            temperature: float=None,
            mean_fan_in: float=None,
            max_gates: jnp.ndarray=None,
            min_gates: jnp.ndarray=None,
            num_wires: int=None) -> float:
            """
            calculates the loss including convolutional layers

            Parameters
            network - [weights, weights_conv], where weights are the dense layers,
            and weights_conv are the convolutional
            inputs - all of the inputs (training xs)
            output - all of the outputs (training labels or ys)
            
            Returns
            loss
            """
            pred = jax.vmap(feed_forward_conv, in_axes=(0, None, 0))(
                inputs, network[1], scaled)
            pred = pred.reshape(pred.shape[0], -1)
            inputs = inputs.reshape(inputs.shape[0], -1)
            inputs = jnp.concatenate([inputs, pred], axis=1)
            return loss(network[0], pred, output, max_fan_in=max_fan_in,
                        temperature=temperature, mean_fan_in=mean_fan_in,
                        max_gates=max_gates, min_gates=min_gates,
                        num_wires=num_wires)

        grad_conv = jax.jit(jax.grad(loss_conv))

    @partial(jax.jit, static_argnames=("use_surr", "max_fan_in_penalty_coeff"))
    def test(weights: Network,
            inputs: jnp.ndarray,
            output: jnp.ndarray,
            use_surr: bool=False,
            surr_arr: List[jnp.ndarray]=[],
            max_fan_in_penalty_coeff: int=0,
            max_fan_in: int=0) -> bool:
        """
        is true iff the network is 100% accurate

        Parameters
        weights - the network
        inputs - jnp array of the inputs we're testing
        output - jnp array of the outputs we're testing
        use_surr - boolean telling us if we're using surrogate bits
        surr_arr - data structure of how to calculate surrogate bits
        
        Returns
        if the network was 100% accurate
        """
        pred = jax.vmap(feed_forward, in_axes=(0, None, None, None, None))(
            inputs, weights, "disc", use_surr, surr_arr)
        if max_fan_in_penalty_coeff:
            return ((1 - max_fan_in_penalty_disc(weights, max_fan_in))
                    * jnp.all(pred==output))
        return jnp.all(pred==output)

    @partial(jax.jit, static_argnames=("use_surr", "max_fan_in_penalty_coeff"))
    def test_rand(weights: Network,
                  inputs: jnp.ndarray,
                  output: jnp.ndarray,
                  use_surr: bool=False,
                  surr_arr: List[jnp.ndarray]=[],
                  max_fan_in_penalty_coeff: int=0,
                  max_fan_in: int=0) -> bool:
        """
        is true iff the network is 100% accurate

        Parameters
        weights - the network
        inputs - jnp array of the inputs we're testing
        output - jnp array of the outputs we're testing
        use_surr - boolean telling us if we're using surrogate bits
        surr_arr - data structure of how to calculate surrogate bits
        
        Returns
        if the network was 100% accurate
        """
        pred = jax.vmap(feed_forward, in_axes=(0, None, None, None, None))(
            inputs, weights, "rand", use_surr, surr_arr)
        if max_fan_in_penalty_coeff:
            return ((1 - max_fan_in_penalty_rand(weights, max_fan_in))
                    * jnp.all(pred==output))
        return jnp.all(pred==output)

    @partial(jax.jit, static_argnames=("skew_towards_falses", "use_surr"))
    def acc(weights: Network,
            inputs: jnp.ndarray,
            output: jnp.ndarray,
            use_surr: bool=False,
            surr_arr: List[jnp.ndarray]=[],
            skew_towards_falses=False
            ) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
        """current_max_fan_in
        calculates the accuracy, and also the masks used in the loss function

        Parameters
        weights - the network
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
        pred = jax.vmap(feed_forward, in_axes=(0, None, None, None, None))(
            inputs, weights, "disc", use_surr, surr_arr)
        pred = (pred == output)
        pred = jnp.sum(pred, axis=1)
        if skew_towards_falses:
            trues = jnp.where(pred == outs)
            falses = jnp.where(pred < outs)
            return jnp.sum(pred)/((2**(ins))*(outs)), trues[0], falses[0]
        return jnp.sum(pred)/((2**(ins))*(outs)), None, None

    @partial(jax.jit, static_argnames=("skew_towards_falses", "use_surr"))
    def rand_acc(weights: Network,
            inputs: jnp.ndarray,
            output: jnp.ndarray,
            use_surr: bool=False,
            surr_arr: List[jnp.ndarray]=[],
            skew_towards_falses=False
            ) -> float:
        """
        calculates the accuracy, and also the masks used in the loss function

        Parameters
        weights - the network
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
        pred = jax.vmap(feed_forward, in_axes=(0, None, None, None, None))(
            inputs, weights, "rand", use_surr, surr_arr)
        pred = (pred == output)
        pred = jnp.sum(pred, axis=1)
        return jnp.sum(pred)/((2**(ins))*(outs))

    loss_kwargs = {"max_fan_in": max_fan_in,
                "temperature": temperature,
                "mean_fan_in": mean_fan_in,
                "max_gates": max_gates,
                "min_gates": min_gates,
                "num_wires": num_wires,
                "use_surr": use_surr,
                "surr_arr": surr_arr}

    loss_conv_kwargs = {"max_fan_in": max_fan_in,
                        "temperature": temperature,
                        "mean_fan_in": mean_fan_in,
                        "max_gates": max_gates,
                        "min_gates": min_gates,
                        "num_wires": num_wires,}

    def filtered_mean(x: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        """
        Computes the mean of x, excluding elements that are 0, inf, -inf or NaN
        
        Parameters:
        x – the input array
        
        Returns:
        A scalar jnp.ndarray representing the filtered mean, or NaN if no valid
        elements.
        """
        total = 0
        count = 0
        for layer in x:
            valid_mask = ~(jnp.isnan(layer) | jnp.isinf(layer) | (layer == 0))
            total += jnp.sum(jnp.abs(layer[valid_mask]))
            count += jnp.sum(valid_mask)
        if count == 0:
            return jnp.nan
        return total/count

    if add_img_or_custom == 'i':

        @jax.jit
        def acc_conv(network: List[Network],
                    inputs: jnp.ndarray,
                    output: jnp.ndarray,
                    scaled: List[jnp.ndarray]=None,
                    ) -> float:
            """
            calculates the accuracy for images

            Parameters
            network - [weights, weights_conv], where weights are the dense layers,
            and weights_conv are the convolutional
            inputs - jnp array of the inputs we're testing
            output - jnp array of the outputs we're testing
            
            Returns
            accuracy - the accuracy (may be specifically the testing accuracy)
            """
            if not (convs is None):
                conv_outputs = jax.vmap(feed_forward_conv, in_axes=(0, None, 0, None))(
                    inputs, network[1], scaled, "disc")
                conv_outputs = conv_outputs.reshape(conv_outputs.shape[0], -1)
            inputs = inputs.reshape(inputs.shape[0], -1)
            inputs = jnp.concatenate([inputs, conv_outputs], axis=1)
            pred = jax.vmap(feed_forward, in_axes=(0, None, None))(
                inputs, network[0], "disc")
            result = jax.vmap(image_util.evaluate)(pred, output)
            return jnp.sum(result)/result.size

        @jax.jit
        def rand_acc_conv(network: List[Network],
                    inputs: jnp.ndarray,
                    output: jnp.ndarray,
                    scaled: List[jnp.ndarray]=None,
                    ) -> float:
            """
            calculates the accuracy for images

            Parameters
            network - [weights, weights_conv], where weights are the dense layers,
            and weights_conv are the convolutional
            inputs - jnp array of the inputs we're testing
            output - jnp array of the outputs we're testing
            
            Returns
            accuracy - the accuracy (may be specifically the testing accuracy)
            """
            if not (convs is None):
                conv_outputs = jax.vmap(feed_forward_conv, in_axes=(0, None, 0, None))(
                    inputs, network[1], scaled, "rand")
                conv_outputs = conv_outputs.reshape(conv_outputs.shape[0], -1)
            inputs = inputs.reshape(inputs.shape[0], -1)
            inputs = jnp.concatenate([inputs, conv_outputs], axis=1)
            pred = jax.vmap(feed_forward, in_axes=(0, None, None))(
                inputs, network[0], "rand")
            result = jax.vmap(image_util.evaluate)(pred, output)
            return jnp.sum(result)/result.size

    batches = config["batches"]
    batch_size = num_ins//batches

    boundary_jump = 5*(max(10//batches,1)**2)*batch_size*config["schedule_scale"]
    lr_multiplier = batch_size**0.5

    print("Learning:\n", output, "\nwith arch:", true_arch)
    start_time = time.time()

    weights = initialise(arch, true_arch, dense_distribution, dense_sigma, dense_k)
    if add_img_or_custom == 'i':
        weights_conv = initialise_conv(convs, conv_distribution, conv_sigma, conv_k)

    schedule_dense = optax.join_schedules(
        schedules = [optax.constant_schedule(
            lr*lr_multiplier) for lr in config["lr_dense"]],
        boundaries=[(i+1)**2*boundary_jump for i in range(1)]
    )
    optimizer_dense = optax.adam(learning_rate=schedule_dense)

    if add_img_or_custom == 'i':
        lr_convs = config["lr_dense"].copy()
        if convs:
            dense_mean = 0
            convs_mean = 0
            for batch in range(batches):
                gradients = grad_conv([weights, weights_conv],
                                    inputs[batch*batch_size:(batch+1)*batch_size],
                                    output[batch*batch_size:(batch+1)*batch_size],
                                    [imgs[batch*batch_size:(batch+1)*batch_size]
                                    for imgs in scaled_train_imgs],
                                    **loss_conv_kwargs)
                dense_mean += filtered_mean(gradients[0])
                convs_mean += filtered_mean(gradients[1])
            dense_mean /= batches
            convs_mean /= batches
            print(dense_mean, convs_mean)
            scaling_factor = dense_mean/convs_mean
            lr_convs = [x*scaling_factor for x in lr_convs]
            print(config["lr_dense"])
            print(lr_convs)
        schedule_conv = optax.join_schedules(
            schedules = [optax.constant_schedule(
                lr*lr_multiplier) for lr in lr_convs],
            boundaries=[(i+1)**2*boundary_jump for i in range(1)]
        )
        optimizer_conv = optax.adam(learning_rate=schedule_conv)

    opt_state_dense = optimizer_dense.init(weights)
    if add_img_or_custom =='i' and convs:
        opt_state_conv = optimizer_conv.init(weights_conv)

    init_time = time.time()
    print("Took", init_time-start_time, "seconds to initialise.")

    print([layer.shape for layer in weights])
    if add_img_or_custom == 'i' and convs:
        print([layer.shape for layer in weights_conv])

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
            # print(sliced_args)
            # print(sliced_kwargs)
            output += func(*sliced_args, **sliced_kwargs)
        return output/batches

    if add_img_or_custom == 'i':
        accs = []
        losses = []
        rand_accs = []
        accuracy = batch_comp(
            partial(acc_conv, network=[weights, weights_conv]),
            batch_size, x_test.shape[0]//batch_size,
            inputs=x_test, output=y_test, scaled=scaled_test_imgs)
        rand_accuracy = batch_comp(
            partial(rand_acc_conv, network=[weights, weights_conv]),
            batch_size, x_test.shape[0]//batch_size,
            inputs=x_test, output=y_test, scaled=scaled_test_imgs)
        new_loss = batch_comp(
            partial(loss_conv, network=[weights, weights_conv], **loss_conv_kwargs),
            batch_size, batches,
            inputs=inputs, output=output, scaled=scaled_train_imgs)
        accs.append(accuracy)
        rand_accs.append(rand_accuracy)
        losses.append(new_loss)
        print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}, Random accuracy: {round(100*float(rand_accuracy),2)}%")
        print(gate_usage_by_layer(weights, "cont"))
        print(gate_usage_by_layer(weights, "disc"))
        print(gate_usage_by_layer(weights, "rand"))
        print(max_fan_in_penalty(weights, 0, temperature),
              max_fan_in_penalty_disc(weights, 0),
              max_fan_in_penalty(weights, max_fan_in, temperature),
              max_fan_in_penalty_disc(weights, max_fan_in))
        print(mean_fan_in_penalty(weights, 0, temperature))
    else:
        accuracy = acc(weights, inputs, output, use_surr, surr_arr, False)[0]
        rand_accuracy = rand_acc(weights, inputs, output, use_surr, surr_arr, False)
        new_loss = loss(weights, inputs, output, **loss_kwargs)
        print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}, Random accuracy: {round(100*float(rand_accuracy),2)}%")
        print(gate_usage_by_layer(weights, "cont"))
        print(gate_usage_by_layer(weights, "disc"))
        print(gate_usage_by_layer(weights, "rand"))
        print(max_fan_in_penalty(weights, 0, temperature),
              max_fan_in_penalty_disc(weights, 0),
              max_fan_in_penalty(weights, max_fan_in, temperature),
              max_fan_in_penalty_disc(weights, max_fan_in))
        print(mean_fan_in_penalty(weights, 0, temperature))

    cont = True
    iters = 0
    start_run_time = time.time()
    outputted = False
    while cont:
        iters += 1
        for _ in range(max(10//batches, 1)):
            if batches > 1:
                key = jax.random.PRNGKey(random.randint(0, 10000))
                shuffled_indices = jax.random.permutation(key, inputs.shape[0])
                inputs = inputs[shuffled_indices]
                output = output[shuffled_indices]
                if add_img_or_custom == 'i' and convs:
                    scaled_train_imgs = [imgs[shuffled_indices]
                                        for imgs in scaled_train_imgs]
            for batch in range(batches):
                if add_img_or_custom == 'i':
                    gradients = grad_conv([weights, weights_conv],
                                        inputs[batch*batch_size:(batch+1)*batch_size],
                                        output[batch*batch_size:(batch+1)*batch_size],
                                        [imgs[batch*batch_size:(batch+1)*batch_size]
                                        for imgs in scaled_train_imgs],
                                        **loss_conv_kwargs)
                    update, opt_state_dense = optimizer_dense.update(
                        gradients[0], opt_state_dense, weights)
                    weights = optax.apply_updates(weights, update)
                    if convs:
                        update, opt_state_conv = optimizer_conv.update(
                            gradients[1], opt_state_conv, weights_conv)
                        weights_conv = optax.apply_updates(weights_conv, update)
                else:
                    gradients = grad(weights,
                                    inputs[batch*batch_size:(batch+1)*batch_size],
                                    output[batch*batch_size:(batch+1)*batch_size],
                                    **loss_kwargs)
                    updates, opt_state_dense = optimizer_dense.update(
                        gradients, opt_state_dense, weights)
                    weights = optax.apply_updates(weights, updates)
            if time.time() - start_run_time > config["timeout"] * 60 and not outputted:
                if add_img_or_custom == 'i':
                    accuracy = batch_comp(
                        partial(acc_conv, network=[weights, weights_conv]),
                        batch_size, x_test.shape[0]//batch_size,
                        inputs=x_test, output=y_test, scaled=scaled_test_imgs)
                    rand_accuracy = batch_comp(
                        partial(rand_acc_conv, network=[weights, weights_conv]),
                        batch_size, x_test.shape[0]//batch_size,
                        inputs=x_test, output=y_test, scaled=scaled_test_imgs)
                    new_loss = batch_comp(
                        partial(loss_conv, network=[weights, weights_conv],
                                **loss_conv_kwargs),
                        batch_size, batches,
                        inputs=inputs, output=output, scaled=scaled_train_imgs)
                    accs.append(float(accuracy))
                    rand_accs.append(float(rand_accuracy))
                    losses.append(float(new_loss))
                    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}, Random accuracy: {round(100*float(rand_accuracy),2)}%")
                    print(gate_usage_by_layer(weights, "cont"))
                    gate_usage_disc = gate_usage_by_layer(weights, "disc")
                    print(gate_usage_disc)
                    gate_usage_rand = gate_usage_by_layer(weights, "rand")
                    print(gate_usage_rand)
                    max_fan = max_fan_in_penalty_disc(weights, 0)
                    print(max_fan_in_penalty(weights, 0, temperature),
                          max_fan,
                          max_fan_in_penalty(weights, max_fan_in, temperature),
                          max_fan_in_penalty_disc(weights, max_fan_in))
                    print(mean_fan_in_penalty(weights, 0, temperature))
                    with open(config["output_file"], "a") as f:
                        for pair in variables.items():
                            f.write(str(pair)+'\n')
                        f.write(f"Accuracies: {accs}\n")
                        f.write(f"Random accuracies: {rand_accs}\n")
                        f.write(f"Losses: {losses}\n")
                        f.write(f"Final gate usage disc: {gate_usage_disc}\n")
                        f.write(f"Final gate usage rand: {gate_usage_rand}\n")
                        f.write(f"Final max fan-in: {max_fan}\n")
                    return
                else:
                    accuracy = acc(weights, inputs, output,
                                use_surr, surr_arr, False)[0]
                    rand_accuracy = rand_acc(weights, inputs, output,
                                            use_surr, surr_arr, False)
                    new_loss = loss(weights, inputs, output, **loss_kwargs)
                    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}, Random accuracy: {round(100*float(rand_accuracy),2)}%")
                    print(gate_usage_by_layer(weights, "cont"))
                    gate_usage_disc = gate_usage_by_layer(weights, "disc")
                    print(gate_usage_disc)
                    print(gate_usage_by_layer(weights, "rand"))
                    max_fan = max_fan_in_penalty_disc(weights, 0)
                    print(max_fan_in_penalty(weights, 0, temperature),
                          max_fan,
                          max_fan_in_penalty(weights, max_fan_in, temperature),
                          max_fan_in_penalty_disc(weights, max_fan_in))
                    print(mean_fan_in_penalty(weights, 0, temperature))
                    if (accuracy >= 0.99 and 
                        max_fan_in_penalty_disc(weights, max_fan_in) == 0) or (
                        rand_accuracy >= 0.99 and
                        max_fan_in_penalty_rand(weights, max_fan_in) == 0):
                        print("Should timeout but will continue")
                        outputted = True
                        if (accuracy >= 0.998 and 
                            max_fan_in_penalty_disc(weights, max_fan_in) == 0):
                            print("Trying step discretisation")
                            [print(circ) for circ in (output_circuit(weights, True, True))]
                        else:
                            print("Trying random discretisation")
                            [print(circ) for circ in (output_circuit(weights, True, True, "rand"))]
                    else:
                        with open(config["output_file"], "a") as f:
                            for pair in variables.items():
                                f.write(str(pair)+'\n')
                            f.write(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}, Random accuracy: {round(100*float(rand_accuracy),2)}%\n")
                            f.write("Circuit output for at the final stage\n")
                            if rand_accuracy == 1:
                                [print(circ) for circ in (output_circuit(weights, True, True, "rand"))]
                            elif accuracy == 1:
                                [print(circ) for circ in (output_circuit(weights, True, True))]
                        print("Timeout")
                        return
        if add_img_or_custom != 'i':
            if test(weights, inputs, output, use_surr, surr_arr, max_fan_in_penalty_coeff, max_fan_in):
                cont = False
            # elif test_rand(weights, inputs, output, use_surr, surr_arr, max_fan_in_penalty_coeff, max_fan_in):
            #     cont = 0
        if cont:
            if add_img_or_custom == 'i':
                accuracy = batch_comp(
                    partial(acc_conv, network=[weights, weights_conv]),
                    batch_size, x_test.shape[0]//batch_size,
                    inputs=x_test, output=y_test, scaled=scaled_test_imgs)
                rand_accuracy = batch_comp(
                    partial(rand_acc_conv, network=[weights, weights_conv]),
                    batch_size, x_test.shape[0]//batch_size,
                    inputs=x_test, output=y_test, scaled=scaled_test_imgs)
                new_loss = batch_comp(
                    partial(loss_conv, network=[weights, weights_conv], **loss_conv_kwargs),
                    batch_size, batches,
                    inputs=inputs, output=output, scaled=scaled_train_imgs)
                accs.append(float(accuracy))
                rand_accs.append(float(rand_accuracy))
                losses.append(float(new_loss))
            if iters == max(10//batches, 1):
                if add_img_or_custom == 'i':
                    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}, Random accuracy: {round(100*float(rand_accuracy),2)}%")
                    print(gate_usage_by_layer(weights, "cont"))
                    print(gate_usage_by_layer(weights, "disc"))
                    print(gate_usage_by_layer(weights, "rand"))
                    print(max_fan_in_penalty(weights, 0, temperature),
                          max_fan_in_penalty_disc(weights, 0),
                          max_fan_in_penalty(weights, max_fan_in, temperature),
                          max_fan_in_penalty_disc(weights, max_fan_in))
                    print(mean_fan_in_penalty(weights, 0, temperature))
                else:
                    accuracy = acc(weights, inputs, output,
                                use_surr, surr_arr, False)[0]
                    rand_accuracy = rand_acc(weights, inputs, output,
                                            use_surr, surr_arr, False)
                    new_loss = loss(weights, inputs, output, **loss_kwargs)
                    print(f"Accuracy: {round(100*float(accuracy),2)}%, Loss: {round(float(new_loss),dps)}, Random accuracy: {round(100*float(rand_accuracy),2)}%")
                    print(gate_usage_by_layer(weights, "cont"))
                    print(gate_usage_by_layer(weights, "disc"))
                    print(gate_usage_by_layer(weights, "rand"))
                    print(max_fan_in_penalty(weights, 0, temperature),
                          max_fan_in_penalty_disc(weights, 0),
                          max_fan_in_penalty(weights, max_fan_in, temperature),
                          max_fan_in_penalty_disc(weights, max_fan_in))
                    print(mean_fan_in_penalty(weights, 0, temperature))
                    if outputted and not (accuracy >= 0.99 and 
                        max_fan_in_penalty_disc(weights, max_fan_in) == 0) or (
                        rand_accuracy >= 0.99 and
                        max_fan_in_penalty_rand(weights, max_fan_in) == 0):
                        cont = False
                iters = 0
    end_time = time.time()
    print("Took", end_time-start_run_time, "seconds to train.")
    if add_img_or_custom != 'i':
        print(max_fan_in_penalty(weights, 0, temperature),
                max_fan_in_penalty_disc(weights, 0),
                max_fan_in_penalty(weights, max_fan_in, temperature),
                max_fan_in_penalty_disc(weights, max_fan_in),
                max_fan_in_penalty_rand(weights, max_fan_in))
        if cont is 0:
            print("Trying random discretisation")
            [print(circ) for circ in (output_circuit(weights, True, True, "rand"))]
        else:
            print("Trying step discretisation")
            [print(circ) for circ in (output_circuit(weights, True, True))]
    return

with open("set-up.yaml", "r") as f:
    config = yaml.safe_load(f)
with open(config["output_file"], "w") as f:
    f.write(f"New test:\n")
true_start = time.time()
sigmas = {"beta_sampler": [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
          "normal_sampler1": [1, 2, 3, 4, 5, 6],
          "normal_sampler2": [0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 2.5]}
ALL_SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
            0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
ALL_KS = [1.0, 1.0, 1.0, 0.995, 0.99, 0.98, 0.97, 0.955, 0.94, 0.92, 0.91,
        0.9, 0.85, 0.75, 0.65, 0.5, 0.39, 0.32, 0.27, 0.23,
        0.205, 0.18, 0.17, 0.155, 0.14, 0.13, 0.12, 0.11]
ks = {s:k for (s,k) in zip(ALL_SIGMAS, ALL_KS)}
distributions = ["beta_sampler", "normal_sampler1", "normal_sampler2"]

"""
archs = [[1024], [1024, 768], [1024, 768, 512], [1024, 768, 512, 256]]
pools = [[], [[3, 1, "max"]], [[3, 1, "min"]], [[3, 1, "max"], [3, 1, "min"]]]
mgms = [0, 0.25, 0.5, 0.75, 1]
max_fans = [0, 32, 64, 96, 128]
for _ in range(15):
    arch = random.choice(archs)
    pf = random.choice(pools)
    max_fan = random.choice(max_fans)
    min_gates = [1568] + arch.copy() + [10]
    min_gates = [round(random.choice(mgms) * layer) for layer in min_gates]
    mgpc = 1 if sum(min_gates) else 0
    mfc = 1 if max_fan else 0
    run_start = time.time()
    run_test({"min_gates_used_penalty_coeff": mgpc,
              "min_gates": min_gates,
              "pool_filters": pf,
              "architecture": arch,
              "max_fan_in_penalty_coeff": mfc,
              "max_fan_in": max_fan})
    run_end = time.time()
    with open("set-up.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(config["output_file"], "a") as f:
        f.write(f"Total time for test: {run_end - run_start} seconds.\n")
true_end = time.time()
with open("set-up.yaml", "r") as f:
    config = yaml.safe_load(f)
with open(config["output_file"], "a") as f:
    f.write(f"Total time for 20 tests: {true_end - true_start} seconds.\n")
"""

archs = [[2048], [1024, 768, 512, 256]]
arch = [1024, 768, 512, 256]
pools = [[], [[3, 1, "max"], [3, 1, "min"]]]
pf = [[3, 1, "max"], [3, 1, "min"]]
mgms = [0, 0.5, 1]
# max_fans = [0, 32, 64, 96, 128]
for mgm in mgms:
    min_gates = [1568] + arch.copy() + [10]
    min_gates = [round(mgm * layer) for layer in min_gates]
    mgpc = 1 if mgm else 0
    run_start = time.time()
    run_test({"min_gates_used_penalty_coeff": mgpc,
            "min_gates": min_gates,
            "pool_filters": pf,
            "architecture": arch})
    run_end = time.time()
    with open("set-up.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(config["output_file"], "a") as f:
        f.write(f"Total time for test: {run_end - run_start} seconds.\n")
true_end = time.time()
with open("set-up.yaml", "r") as f:
    config = yaml.safe_load(f)
with open(config["output_file"], "a") as f:
    f.write(f"Total time for 20 tests: {true_end - true_start} seconds.\n")