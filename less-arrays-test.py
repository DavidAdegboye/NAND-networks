import jax
import jax.numpy as jnp
import optax #type: ignore
import random
import math
from typing import List, Tuple, Set, Union

# in some sense a neuron is a list of layers also, which 
# can cause some logical bugs. For example, f is calculating
# values over a layer

# want neuron to be an array of arrays
Neuron = List[jnp.ndarray]
Layer = List[Neuron]
Network = List[Layer]

##jax.config.update("jax_traceback_filtering", "off")

width = 50
hidden = 6
bits = 6
def denary_to_binary_array(number: jnp.ndarray, bits: int=bits*2) -> jnp.ndarray:
    return jnp.array([(jnp.right_shift(number, bits - 1 - i) & 1) for i in range(bits)], dtype=jnp.int32)

def get_output(number: jnp.ndarray) -> jnp.ndarray:
    return denary_to_binary_array(number//(2**bits) + number%(2**bits), bits=bits+1)

inputs = jax.vmap(denary_to_binary_array)(jnp.arange(2**(bits*2)))
output = jax.vmap(get_output)(jnp.arange(2**(bits*2)))
# arch = [4,19,15,10,7,5,3]
# arch = [3,4,3,3,3,2]
# arch = [2,1,2,2]
arch = [bits*2] + [width] * hidden + [bits+1]

def f(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    # x would be all of the inputs coming in from a certain layer
    # w would be all of the weights for inputs to that layer to a given NAND gate
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w))
f = jax.jit(f)

def f_disc(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return jnp.prod(jnp.where(w>0, x, 1)) 
f_disc = jax.jit(f_disc)

def forward(weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    # the forward pass for an arbitrary neuron. 1 - the product of all the fs
    # to use vmap, I include some padding that doesn't affect the value.
    # x=1, w=0, since f(1,0)=1, so it wouldn't affect the result
    # after the product.
    return 1 - jnp.prod(jax.vmap(f)(xs, weights))
forward = jax.jit(forward)

def forward_disc(weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    return 1 - jnp.prod(jax.vmap(f_disc)(xs, weights))
forward_disc = jax.jit(forward_disc)

def output_circuit(neurons: jnp.ndarray, verbose=False) -> List[str]:
    circuits = [chr(ord('A')+i) for i in range(arch[0])]
    gates:List[List[List[Union[str,Tuple[int,int]]]]] = [[[] for _ in range(arch[0])]]
    c2i = dict([(x,i) for i,x in enumerate(circuits)])
    indices = dict([(i,i) for i in range(arch[0])])
    index2gate = dict([(i, (0,i)) for i in range(arch[0])])
    empties = []
    added = arch[0] - 1
    for layer_i in range(i_1):
        gates.append([])
        gate_i1 = layer_i+1
        gate_i2 = 0
        for neuron_i in range(len(shapes[layer_i])):
            i = 0
            connected: Set[Tuple[int, str]] = set()
            for inner_layer_i in range(len(shapes[layer_i][neuron_i])):
                for weight_i in range(shapes[layer_i][neuron_i][inner_layer_i]):
                    if neurons[layer_i,neuron_i,inner_layer_i,weight_i] > 0 and indices[i] not in empties:
                        connected.add((indices[i], circuits[indices[i]]))
                    i += 1
            added += 1
            connected = list(connected) #type: ignore
            if not connected:
                empties.append(added)
                indices[added] = added
                circuits.append('_')
            else:
                if len(connected) == 1:
                    node = '¬' + connected[0][1] #type: ignore
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
    if verbose:
        print(c2i)
        print(indices)
        print(circuits)
        print(gates)
    return circuits[-arch[-1]:]
##output_circuit = jax.jit(output_circuit)

def feed_forward(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
    for layer_i in range(i_1-1):
        next = jax.vmap(forward, in_axes=(0, None))(neurons[layer_i], xs)
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    return jax.vmap(forward, in_axes=(0, None))(neurons[i_1-1], xs)[:len(output[0])]
feed_forward = jax.jit(feed_forward)

def feed_forward_disc(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
    for layer_i in range(i_1-1):
        next = jax.vmap(forward_disc, in_axes=(0, None))(neurons[layer_i], xs)
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    return jax.vmap(forward_disc, in_axes=(0, None))(neurons[i_1-1], xs)[:len(output[0])]
feed_forward_disc = jax.jit(feed_forward_disc)

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
    n = sum(arch[:layer])
    mu = -jnp.log(n-1)/k
    for i in range(layer):
        inner_layer = sigma * jax.random.normal(jax.random.key(key), (arch[i])) + mu #type: ignore
        weights = weights.at[i].set(jnp.pad(inner_layer, (0, i_4-arch[i]), mode="constant", constant_values=-jnp.inf))
        key = random.randint(0, 10000)
    return weights

def initialise(arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    neurons = []
    for i1 in range(1, len(arch)):
        layer = jnp.ones((arch[i1], i1, i_4))
        for i2 in range(arch[i1]):
            layer = layer.at[i2].set(get_weights(i1, arch, sigma, k))
        neurons.append(layer)
    return neurons

def get_shapes(arch: List[int]) -> Tuple[Network, int]:
    shapes: Network = []
    total = 0
    for layer in range(1, len(arch)):
        shapes.append([])
        for _ in range(arch[layer]):
            shapes[-1].append(arch[:layer].copy())
            total += sum(arch[:layer])
    return shapes, total

def get_l2(neurons: Network) -> float:
    s = 0
    for layer in neurons:
        s += jnp.sum(1-jax.nn.sigmoid(jnp.absolute(layer))) # type: ignore
    return s/total
get_l2 = jax.jit(get_l2)

epsilon = 1e-7
l2_coeff = 0.01
def loss(neurons : Network) -> float:
    pred = jax.vmap(feed_forward, in_axes=(0, None))(inputs, neurons) # type: ignore
    pred = jnp.clip(pred, epsilon, 1-epsilon)
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    l2 = get_l2(neurons)
    return l1 + l2_coeff * l2 # type: ignore
loss = jax.jit(loss)

def test(neurons: jnp.ndarray) -> jnp.ndarray:
    pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neurons)
    return jnp.all(pred==output)
# test = jax.jit(test)
        
## TODO : add jit, wrap loss function in jax value and grad, vmap, jax.debug.print
## every n iterations, discretize the circuit, test and potentially stop if 100% accurate
## try on adders, other 2 bit logic gates.

def solver_init(neurons: jnp.ndarray, opt_state, pass_through=False):
    return jax.lax.cond(pass_through, lambda _:opt_state, lambda _:solver.init(neurons), operand=None)

print("Learning:", output, "with arch:", arch)
key = random.randint(0, 10000)

i_1 = len(arch) - 1
i_2 = max(arch[1:])
i_3 = i_1
i_4 = max(arch)
shapes, total = get_shapes(arch)
sigmas = jnp.array([0.01, 0.1, 0.25, 0.3, 0.4, 0.45, 0.5, 0.53, 0.6, 0.62, 0.7, 0.75, 0.8, 1.0])
#some extra sigmas: 2.0, 3.0, 5.0, 10.0
ks = jnp.array([1.0, 1.0, 0.99, 0.98, 0.97, 0.96, 0.955, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.85])
#some extra ks: 0.65, 0.5, 0.32, 0.17
n = sigmas.shape[0]
neuronss = [initialise(arch, sigmas[i], ks[i]) for i in range(n)]
solver = optax.adam(learning_rate=0.003)
opt_states = [solver.init(neurons) for neurons in neuronss]
# print(opt_states[0])
cont = True
old_losses = [round(float(loss(neuronss[index])),3) for index in range(n)]
print("Losses:", old_losses)
grad = jax.grad(loss)
iters = 0

def update_fun(opt_state: optax.OptState, neurons: jnp.ndarray) -> Tuple[optax.OptState, jnp.ndarray]:
    updates, new_state = solver.update(grad(neurons), opt_state, neurons)
    new_neurons = optax.apply_updates(neurons, updates)
    return new_state, new_neurons

while cont:
    iters += 1
    for _ in range(10):
        for i in range(n):
            opt_states[i], neuronss[i] = update_fun(opt_states[i], neuronss[i])
    for index, neurons in enumerate(neuronss):
        if test(neurons):
            print(index)
            jax.debug.print("neurons:{}", neurons)
            print(jax.vmap(feed_forward_disc_print, in_axes=(0, None))(inputs, neuronss[index]))
            final_index = index
            cont = False
    if cont:
        new_losses = [round(float(loss(neuronss[index])),3) for index in range(n)]
        for i in range(n):
            if new_losses[i] >= old_losses[i] or math.isnan(new_losses[i]):
                print("Restarting", i, "with new random weights, stuck in local minima", new_losses[i], old_losses[i])
                # print(output_circuit(neuronss[i]))
                # print(jax.vmap(feed_forward_disc, in_axes=(0, None))(inputs, neuronss[i]))
                neuronss[i] = initialise(arch, sigmas[i], ks[i]) 
                opt_states[i] = solver.init(neuronss[i])
                new_losses[i] = round(float(loss(neuronss[i])),3) 
                old_losses[i] = 100
                print(new_losses[i], old_losses[i])
            else:
                old_losses[i] = new_losses[i]
        if iters == 10:
            print("Losses:", new_losses)
            iters = 0
print("Learnt:", output, "with arch:", arch)
print(output_circuit(neuronss[final_index], verbose=True))
