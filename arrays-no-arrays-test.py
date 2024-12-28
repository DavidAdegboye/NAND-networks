import jax
import jax.numpy as jnp
import optax #type: ignore
import random
import math
from typing import List, Tuple, Set, Union

import timeit

# in some sense a neuron is a list of layers also, which 
# can cause some logical bugs. For example, f is calculating
# values over a layer

# want neuron to be an array of arrays
Neuron = List[jnp.ndarray]
Layer = List[Neuron]
Network = List[Layer]

##jax.config.update("jax_traceback_filtering", "off")

width = 20
hidden = 3
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

def forward_arrays(weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    # the forward pass for an arbitrary neuron. 1 - the product of all the fs
    # to use vmap, I include some padding that doesn't affect the value.
    # x=1, w=0, since f(1,0)=1, so it wouldn't affect the result
    # after the product.
    return 1 - jnp.prod(jax.vmap(f)(xs, weights))
forward_arrays = jax.jit(forward_arrays)

def feed_forward_arrays(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    xs = jnp.ones((i_3,i_4))
    xs = xs.at[0].set(jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1))
    for layer_i in range(i_1-1):
        xs = xs.at[layer_i+1, :arch[layer_i+1]].set(jax.vmap(forward_arrays, in_axes=(0, None))(neurons[layer_i], xs)[:arch[layer_i+1]])
    return jax.vmap(forward_arrays, in_axes=(0, None))(neurons[i_1-1], xs)[:len(output[0])]
feed_forward_arrays = jax.jit(feed_forward_arrays)

def get_weights_arrays(layer: int, arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    global key
    weights = jnp.ones((i_3,i_4)) * -jnp.inf
    n = sum(arch[:layer])
    mu = -jnp.log(n-1)/k
    for i in range(layer):
        inner_layer = sigma * jax.random.normal(jax.random.key(key), (arch[i])) + mu #type: ignore
        weights = weights.at[i].set(jnp.pad(inner_layer, (0, i_4-arch[i]), mode="constant", constant_values=-jnp.inf))
        key = random.randint(0, 10000)
    return weights

def initialise_arrays(arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    neurons = jnp.ones((i_1, i_2, i_3, i_4))
    for i1 in range(1, len(arch)):
        for i2 in range(arch[i1]):
            neurons = neurons.at[i1-1,i2].set(get_weights_arrays(i1, arch, sigma, k))
        for i2 in range(arch[i1], i_2):
            neurons = neurons.at[i1-1,i2].set(-jnp.inf * jnp.ones((i_3,i_4)))
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

epsilon = 1e-8
l2_coeff = 0.01
def loss_arrays(neurons: jnp.ndarray) -> jnp.ndarray:
    pred = jax.vmap(feed_forward_arrays, in_axes=(0, None))(inputs, neurons) 
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    l2 = jnp.sum(1-jax.nn.sigmoid(jnp.absolute(neurons))) / total
    return l1 + l2_coeff * l2 
loss_arrays = jax.jit(loss_arrays)

def initialise_some_arrays(arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    neurons = []
    for i1 in range(1, len(arch)):
        layer = jnp.ones((arch[i1], i_3, i_4))
        for i2 in range(arch[i1]):
            layer = layer.at[i2].set(get_weights_arrays(i1, arch, sigma, k))
        neurons.append(layer)
    return neurons

def get_l2_some_arrays(neurons: Network) -> float:
    s = 0
    for layer in neurons:
        s += jnp.sum(1-jax.nn.sigmoid(jnp.absolute(layer))) # type: ignore
    return s/total
get_l2_some_arrays = jax.jit(get_l2_some_arrays)

epsilon = 1e-8
l2_coeff = 0.01
def loss_some_arrays(neurons : Network) -> float:
    pred = jax.vmap(feed_forward_arrays, in_axes=(0, None))(inputs, neurons) # type: ignore
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    l2 = get_l2_some_arrays(neurons)
    return l1 + l2_coeff * l2 # type: ignore
loss_some_arrays = jax.jit(loss_some_arrays)

def get_weights_less_arrays(layer: int, arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
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

def initialise_less_arrays(arch: List[int], sigma: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    neurons = []
    for i1 in range(1, len(arch)):
        layer = jnp.ones((arch[i1], i1, i_4))
        for i2 in range(arch[i1]):
            layer = layer.at[i2].set(get_weights_less_arrays(i1, arch, sigma, k))
        neurons.append(layer)
    return neurons

def feed_forward_less_arrays(inputs: jnp.ndarray, neurons: jnp.ndarray) -> jnp.ndarray:
    xs = jnp.array([jnp.pad(inputs,(0, i_4-len(inputs)), mode="constant", constant_values=1)])
    for layer_i in range(i_1-1):
        next = jax.vmap(forward_arrays, in_axes=(0, None))(neurons[layer_i], xs)
        next = jnp.array([jnp.pad(next,(0, i_4-len(next)), mode="constant", constant_values=1)])
        xs = jnp.vstack([xs, next])
    return jax.vmap(forward_arrays, in_axes=(0, None))(neurons[i_1-1], xs)[:len(output[0])]
feed_forward = jax.jit(forward_arrays)

epsilon = 1e-8
l2_coeff = 0.01
def loss_less_arrays(neurons : Network) -> float:
    pred = jax.vmap(feed_forward_less_arrays, in_axes=(0, None))(inputs, neurons) # type: ignore
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    l2 = get_l2_some_arrays(neurons)
    return l1 + l2_coeff * l2 # type: ignore
loss_less_arrays = jax.jit(loss_less_arrays)

i_1 = len(arch) - 1
i_2 = max(arch[1:])
i_3 = i_1
i_4 = max(arch)
shapes, total = get_shapes(arch)

def forward_no_arrays(weights : Neuron, xs : List[jnp.ndarray]) -> float:
    # weights is a List of jnp arrays of different sizes
    # because these jnp arrays have different sizes, it can't be a jnp array
    return 1 - jnp.prod(jnp.array(list(map(f, xs, weights)))) # type: ignore
forward_no_arrays = jax.jit(forward_no_arrays)

def feed_forward_no_arrays(inputs : jnp.ndarray, neurons : Network) -> jnp.ndarray:
    xs = [inputs]
    for layer in neurons:
        layer_outputs = jnp.array([forward_no_arrays(weights, xs) for weights in layer])
        xs.append(layer_outputs)
    return xs[-1]
feed_forward_no_arrays = jax.jit(feed_forward_no_arrays)

key=0
def get_weights_no_arrays(layer : int, arch : List[int]) -> Neuron:
    global key
    weights = []
    n = sum(arch[:layer])
    mu = -jnp.log(n-1)/0.955
    sigma = 0.5
    for i in range(layer):
        weights.append(sigma * jax.random.normal(jax.random.key(key), (arch[i])) + mu) # type: ignore
        key += 1
    return weights

def initialise_no_arrays(arch : List[int]) -> Network:
    neurons : Network = []
    for i in range(1, len(arch)):
        neurons.append([])
        for _ in range(arch[i]):
            weights = get_weights_no_arrays(i, arch)
            neurons[-1].append(weights)
    return neurons

def get_l2_no_arrays(neurons: Network) -> float:
    n = 0
    total = 0
    for layer in neurons:
        for neuron in layer:
            for inner_layer in neuron:
                total += jnp.sum(1-jax.nn.sigmoid(jnp.absolute(inner_layer))) # type: ignore
                n += inner_layer.size
    return total/n
get_l2_no_arrays = jax.jit(get_l2_no_arrays)

epsilon = 1e-8
l2_coeff = 0.01
def loss_no_arrays(neurons : Network) -> float:
    pred = jax.vmap(feed_forward_no_arrays, in_axes=(0, None))(inputs, neurons) # type: ignore
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    l2 = get_l2_no_arrays(neurons)
    l2 = 0
    return l1 + l2_coeff * l2 # type: ignore
loss_no_arrays = jax.jit(loss_no_arrays)

grad_arrays = jax.grad(loss_arrays)
grad_some_arrays = jax.grad(loss_some_arrays)
grad_less_arrays = jax.grad(loss_less_arrays)
grad_no_arrays = jax.grad(loss_no_arrays)

neurons_arrays = initialise_arrays(arch, 0.5, 0.955)
neurons_some_arrays = initialise_some_arrays(arch, 0.5, 0.955)
neurons_less_arrays = initialise_less_arrays(arch, 0.5, 0.955)
neurons_no_arrays = initialise_no_arrays(arch)

solver = optax.adam(learning_rate=0.003)
opt_states_arrays = solver.init(neurons_arrays)
opt_states_some_arrays = solver.init(neurons_some_arrays)
opt_states_less_arrays = solver.init(neurons_less_arrays)
opt_states_no_arrays = solver.init(neurons_no_arrays)

grad_desc_arrays = """
for _ in range(10):
    updates, opt_states_arrays = solver.update(grad_arrays(neurons_arrays), opt_states_arrays, neurons_arrays)
    neurons_arrays = optax.apply_updates(neurons_arrays, updates)
"""

grad_desc_some_arrays = """
for _ in range(10):
    updates, opt_states_some_arrays = solver.update(grad_some_arrays(neurons_some_arrays), opt_states_some_arrays, neurons_some_arrays)
    neurons_some_arrays = optax.apply_updates(neurons_some_arrays, updates)
"""

grad_desc_less_arrays = """
for _ in range(10):
    updates, opt_states_less_arrays = solver.update(grad_less_arrays(neurons_less_arrays), opt_states_less_arrays, neurons_less_arrays)
    neurons_less_arrays = optax.apply_updates(neurons_less_arrays, updates)
"""

grad_desc_no_arrays = """
for _ in range(10):
    updates, opt_states_no_arrays = solver.update(grad_no_arrays(neurons_no_arrays), opt_states_no_arrays, neurons_no_arrays)
    neurons_no_arrays = optax.apply_updates(neurons_no_arrays, updates)
"""

x=feed_forward_arrays(inputs[0], neurons_arrays).block_until_ready()
print(x)
x=feed_forward_arrays(inputs[0], neurons_some_arrays).block_until_ready()
print(x)
x=feed_forward_less_arrays(inputs[0], neurons_less_arrays).block_until_ready()
print(x)
x=feed_forward_no_arrays(inputs[0], neurons_no_arrays).block_until_ready()
print(x)

elapsed_time = timeit.timeit("pred = jax.vmap(feed_forward_arrays, in_axes=(0, None))(inputs, neurons_arrays).block_until_ready()",
    setup="from __main__ import jax, feed_forward_arrays, inputs, neurons_arrays",
    number=1000,
)
print(f"Elapsed time for 1000 feed forwards (arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("pred = jax.vmap(feed_forward_arrays, in_axes=(0, None))(inputs, neurons_some_arrays).block_until_ready()",
    setup="from __main__ import jax, feed_forward_arrays, inputs, neurons_some_arrays",
    number=1000,
)
print(f"Elapsed time for 1000 feed forwards (some arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("pred = jax.vmap(feed_forward_less_arrays, in_axes=(0, None))(inputs, neurons_less_arrays).block_until_ready()",
    setup="from __main__ import jax, feed_forward_less_arrays, inputs, neurons_less_arrays",
    number=1000,
)
print(f"Elapsed time for 1000 feed forwards (less arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("pred = jax.vmap(feed_forward_no_arrays, in_axes=(0, None))(inputs, neurons_no_arrays).block_until_ready()",
    setup="from __main__ import jax, feed_forward_no_arrays, inputs, neurons_no_arrays",
    number=1000,
)
print(f"Elapsed time for 1000 feed forwards (lists): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("loss_arrays(neurons_arrays)",
    setup="from __main__ import loss_arrays, neurons_arrays",
    number=1000,
)
print(f"Elapsed time for 1000 losses (arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("loss_some_arrays(neurons_some_arrays)",
    setup="from __main__ import loss_some_arrays, neurons_some_arrays",
    number=1000,
)
print(f"Elapsed time for 1000 losses (some arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("loss_less_arrays(neurons_less_arrays)",
    setup="from __main__ import loss_less_arrays, neurons_less_arrays",
    number=1000,
)
print(f"Elapsed time for 1000 losses (less arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("loss_no_arrays(neurons_no_arrays)",
    setup="from __main__ import loss_no_arrays, neurons_no_arrays",
    number=1000,
)
print(f"Elapsed time for 1000 losses (lists): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("grad_arrays(neurons_arrays)",
    setup="from __main__ import grad_arrays, neurons_arrays",
    number=100,
)
print(f"Elapsed time for 100 grads (arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("grad_some_arrays(neurons_some_arrays)",
    setup="from __main__ import grad_some_arrays, neurons_some_arrays",
    number=100,
)
print(f"Elapsed time for 100 grads (some arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("grad_less_arrays(neurons_less_arrays)",
    setup="from __main__ import grad_less_arrays, neurons_less_arrays",
    number=100,
)
print(f"Elapsed time for 100 grads (less arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit("grad_no_arrays(neurons_no_arrays)",
    setup="from __main__ import grad_no_arrays, neurons_no_arrays",
    number=100,
)
print(f"Elapsed time for 100 grads (lists): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit(grad_desc_arrays,
    setup="from __main__ import optax, opt_states_arrays, solver, grad_arrays, neurons_arrays",
    number=10,
)
print(f"Elapsed time for 10 grad descents of 10 steps (arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit(grad_desc_some_arrays,
    setup="from __main__ import optax, opt_states_some_arrays, solver, grad_some_arrays, neurons_some_arrays",
    number=10,
)
print(f"Elapsed time for 10 grad descents of 10 steps (some arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit(grad_desc_less_arrays,
    setup="from __main__ import optax, opt_states_less_arrays, solver, grad_less_arrays, neurons_less_arrays",
    number=10,
)
print(f"Elapsed time for 10 grad descents of 10 steps (less arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit(grad_desc_no_arrays,
    setup="from __main__ import optax, opt_states_no_arrays, solver, grad_no_arrays, neurons_no_arrays",
    number=10,
)
print(f"Elapsed time for 10 grad descents of 10 steps (lists): {elapsed_time:.4f} seconds")