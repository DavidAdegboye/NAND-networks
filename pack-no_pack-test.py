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
Neuron = List[int]
Layer = List[Neuron]
Network = List[Layer]

##jax.config.update("jax_traceback_filtering", "off")

width = 30
hidden = 4
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

i_1 = len(arch) - 1
i_2 = max(arch[1:])
i_3 = i_1
i_4 = max(arch)

def f_arrays(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    # x would be all of the inputs coming in from a certain layer
    # w would be all of the weights for inputs to that layer to a given NAND gate
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w))
f_arrays = jax.jit(f_arrays)

def forward_arrays(weights: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    # the forward pass for an arbitrary neuron. 1 - the product of all the fs
    # to use vmap, I include some padding that doesn't affect the value.
    # x=1, w=0, since f(1,0)=1, so it wouldn't affect the result
    # after the product.
    return 1 - jnp.prod(jax.vmap(f_arrays)(xs, weights))
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

def f_no_arrays(x : jnp.ndarray, w : jnp.ndarray) -> jnp.ndarray:
    # x would be all of the inputs coming in from a certain layer
    # w would be all of the weights for inputs to that layer to a given NAND gate
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w)) # type: ignore
f_no_arrays = jax.jit(f_no_arrays)

def forward_no_arrays(weights : Neuron, xs : List[jnp.ndarray]) -> float:
    # weights is a List of jnp arrays of different sizes
    # because these jnp arrays have different sizes, it can't be a jnp array
    return 1 - jnp.prod(jnp.array([f_no_arrays(xi,wi) for xi,wi in zip(xs, weights)])) # type: ignore
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

neurons_arrays = initialise_arrays(arch, 0.5, 0.955)
neurons_no_arrays = initialise_no_arrays(arch)

x=feed_forward_arrays(inputs[0], neurons_arrays).block_until_ready()
print(x)
x=feed_forward_no_arrays(inputs[0], neurons_no_arrays).block_until_ready()
print(x)

elapsed_time = timeit.timeit(
    """
x = feed_forward_arrays(inputs[0], neurons_arrays).block_until_ready()
""",
    setup="""
from __main__ import feed_forward_arrays, inputs, neurons_arrays
""",
    number=1000,
)
print(f"Elapsed time for 1000 executions (arrays): {elapsed_time:.4f} seconds")

elapsed_time = timeit.timeit(
    """
x = feed_forward_no_arrays(inputs[0], neurons_no_arrays).block_until_ready()
""",
    setup="""
from __main__ import feed_forward_no_arrays, inputs, neurons_no_arrays
""",
    number=1000,
)
print(f"Elapsed time for 1000 executions (no arrays): {elapsed_time:.4f} seconds")
