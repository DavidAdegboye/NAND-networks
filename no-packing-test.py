import jax
import jax.numpy as jnp
import optax # type: ignore
import random
import typing
from typing import List, Tuple, Set

# in some sense a neuron is a list of layers also, which 
# can cause some logical bugs. For example, f is calculating
# values over a layer
Neuron = List[jnp.ndarray]
Layer = List[Neuron]
Network = List[Layer]

##jax.config.update("jax_traceback_filtering", "off")

inputs = [[0,0],[0,1],[1,0],[1,1]]
inputs = [jnp.array(x) for x in inputs] # type: ignore
output = jnp.array([jnp.array([0,0]), jnp.array([0,1]), jnp.array([0,1]), jnp.array([1,0])])
arch = [2,1,2,2]

def f(x : jnp.ndarray, w : jnp.ndarray) -> float:
    # x would be all of the inputs coming in from a certain layer
    # w would be all of the weights for inputs to that layer to a given NAND gate
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w)) # type: ignore
f = jax.jit(f)

def f_disc(x : jnp.ndarray, w : jnp.ndarray) -> float:
    return jnp.prod(jnp.where(w>0, x, 1)) # type: ignore
f_disc = jax.jit(f_disc)

## maybe later for efficiency, remove class

def flatten(weights : Neuron) -> List[float]:
    flat = []
    for layer in weights:
        for connection in layer:
            flat.append(connection)
    return flat
flatten = jax.jit(flatten)

def shape(weights : Neuron) -> List[int]:
    return [len(layer) for layer in weights]
shape = jax.jit(shape)

def forward(weights : Neuron, xs : List[jnp.ndarray]) -> float:
    # the forward pass for an arbitrary neuron. 1 - the product of all the fs
    # to use vmap, I would have to include some padding that doesn't affect the value.
    # main candidate would be x=1, w=1, since f(1,1)=1, so it wouldn't affect the result
    # after the product.
    # return 1 - jnp.prod(jax.vmap(f, in_axes=(0,0))(xs, weights))
    return 1 - jnp.prod(jnp.array([f(xi,wi) for xi,wi in zip(xs, weights)])) # type: ignore
forward = jax.jit(forward)

def forward_disc(weights: Neuron, xs : List[jnp.ndarray]) -> float:
    return 1 - jnp.prod(jnp.array([f_disc(xi,wi) for xi,wi in zip(xs, weights)])) # type: ignore
forward_disc = jax.jit(forward_disc)

def output_circuit(neurons: Network, verbose=False) -> List[str]:
    circuits = [chr(ord('A')+i) for i in range(arch[0])]
    c2i = dict({(x,i) for i,x in enumerate(circuits)})
    indices = dict({(i,i) for i in range(arch[0])})
    empties = []
    added = arch[0] - 1
    for layer in neurons:
        for neuron in layer:
            i = 0
            connected : Set[Tuple[int, str]] = set()
            for inner_layer in neuron:
                for weight in inner_layer:
                    if weight > 0 and indices[i] not in empties:
                        connected.add((indices[i], circuits[indices[i]]))
                    i += 1
            added += 1
            connected = list(connected) # type: ignore
            if not connected:
                empties.append(added)
                indices[added] = added
                circuits.append('_')
            else:
                if len(connected) == 1:
                    node = '¬' + connected[0][1] # type: ignore
                    if len(node) > 2:
                        if node[:3] == "¬¬¬":
                            node = node[2:]
                else:
                    node = '¬(' + '.'.join([element[1] for element in connected]) + ')'
                if node in c2i.keys():
                    circuits.append('_')
                    indices[added] = c2i[node]
                else:
                    circuits.append(node)
                    c2i[node] = added
                    indices[added] = added
    if verbose:
        print(i)
        [print(k,v) for k,v in c2i.items()]
        print(neuron)
        print(connected)
    print(circuits[-1])
    return circuits[-arch[-1]:]
##output_circuit = jax.jit(output_circuit)

def feed_forward(inputs : List[int], neurons : Network) -> jnp.ndarray:
    xs = [jnp.array(inputs)]
    for layer in neurons:
        layer_outputs = jnp.array([forward(weights, xs) for weights in layer])
        xs.append(layer_outputs)
    return xs[-1]
feed_forward = jax.jit(feed_forward)

def feed_forward_disc(inputs : List[int], neurons : Network) -> jnp.ndarray:
    xs = [jnp.array(inputs)]
    for layer in neurons:
        layer_outputs = jnp.array([forward_disc(weights, xs) for weights in layer])
        xs.append(layer_outputs)
    return xs[-1]
feed_forward_disc = jax.jit(feed_forward_disc)

def get_weights(layer : int, arch : List[int]) -> Neuron:
    global key
    weights = []
    # layer lists, each with arch[i] elements
    # so this is a 2D list of floats
    # or a 1D list of jnp arrays
    n = sum(arch[:layer])
    mu = -2*jnp.log(n-1)
    sigma = 3
    for i in range(layer):
        weights.append(sigma * jax.random.normal(jax.random.key(key), (arch[i])) + mu) # type: ignore
        key += 1
    return weights

def initialise(arch : List[int]) -> Tuple[Network, List[int]]:
    neurons : Network = []
    weight_counts = []
    for i in range(1, len(arch)):
        neurons.append([])
        for _ in range(arch[i]):
            weights = get_weights(i, arch)
            neurons[-1].append(weights)
            weight_counts.append(sum(shape(weights)))
    return neurons, weight_counts

def get_l2(neurons: Network) -> float:
    n = 0
    total = 0
    for layer in neurons:
        for neuron in layer:
            for inner_layer in neuron:
                total += jnp.sum(1-jax.nn.sigmoid(jnp.absolute(inner_layer))) # type: ignore
                n += inner_layer.size
    return total/n
get_l2 = jax.jit(get_l2)

epsilon = 1e-8
l2_coeff = 0.01
def loss(neurons : Network) -> float:
    pred = []
    for inp in inputs:
        act = feed_forward(inp, neurons)
        act = jnp.clip(act, epsilon, 1-epsilon)
        pred.append(act)
    pred = jnp.array(pred) # type: ignore
    pred_logits = jnp.log(pred) - jnp.log(1-pred) # type: ignore
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output)) # type: ignore
    l2 = get_l2(neurons)
    return l1 + l2_coeff * l2 # type: ignore
loss = jax.jit(loss)

def test(neurons : Network) -> bool:
    pred = []
    for inp in inputs:
        act = feed_forward_disc(inp, neurons)
        pred.append(act)
    return all([all(p==o) for p,o in zip(pred, output)])
# test = jax.jit(test)
        
## TODO : add jit, wrap loss function in jax value and grad, vmap, jax.debug.print
## every n iterations, discretize the circuit, test and potentially stop if 100% accurate
## try on adders, other 2 bit logic gates.

print("Learning:", output, "with arch:", arch)
key = random.randint(0, 10000)

n = 5
neuronss = []
for _ in range(n):
    ne, weight_counts = initialise(arch)
    neuronss.append(ne)
solver = optax.adam(learning_rate=0.003)
opt_states = [solver.init(neurons) for neurons in neuronss]
cont = True
old_losses = []
for index in range(n):
    old_losses.append(round(float(loss(neuronss[index])),3)) # type: ignore
print("Losses:", old_losses)
iters = 0
while cont:
    iters += 1
    for _ in range(10):
        for index in range(n):
            grad = jax.grad(loss)(neuronss[index])
            updates, opt_states[index] = solver.update(grad, opt_states[index], neuronss[index])
            neuronss[index] = optax.apply_updates(neuronss[index], updates)
    for index, neurons in enumerate(neuronss):
        if test(neurons):
            print(index)
            # print(neurons)
            final_index = index
            cont = False
    if cont:
        new_losses = []
        for index in range(n):
            new_losses.append(round(float(loss(neuronss[index])),3)) # type: ignore
        for i in range(n):
            if new_losses[i] >= old_losses[i]:
                print("Restarting", i, "with new random weights, stuck in local minima", new_losses[i], old_losses[i])
                neuronss[i], _ = initialise(arch) # type: ignore
                opt_states[i] = solver.init(neuronss[i])
                new_losses[i] = round(float(loss(neuronss[index])),3) # type: ignore
                old_losses[i] = 100
            else:
                old_losses[i] = new_losses[i]
        if iters == 10:
            print("Losses:", new_losses)
            iters = 0
print("Learnt:", output, "with arch:", arch)
output_circuit(neuronss[final_index])