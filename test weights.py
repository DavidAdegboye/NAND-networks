import tensorflow as tf
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os
from skimage.transform import resize
from typing import List, Tuple
import jax
import jax.numpy as jnp
import optax
import random
import itertools
from typing import List, Tuple, Set, Union
import time
print("Enter what was used for the tests")
import image_class
Network = List[jnp.ndarray]


file_to_load = "weights2 (1).txt"
extra_layers, arch, some_or_less, s, convs, neurons = image_class.load(file_to_load)
outs = arch[-1]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
print(x_train.shape)
print(x_test.shape)
x_train = jnp.array([image_class.preprocess_image(img, s=(s,s)) for img in x_train])
x_test = jnp.array([image_class.preprocess_image(img, s=(s,s)) for img in x_test])
print(x_train.shape)
print(x_test.shape)
x_train = image_class.prep_test(x_train, convs, 60000)
x_test = image_class.prep_test(x_test, convs)
print(x_train.shape)
print(x_test.shape)
y_train_loss = jax.vmap(image_class.preprocess_test)(y_train)
y_test_loss = jax.vmap(image_class.preprocess_test)(y_test)

def add_second_layers(input: jnp.ndarray, min_fan: int, max_fan: int) -> jnp.ndarray:
    # giving the network the second layer for free. Can hypothetically do this n times, although its cost grows exponentially.
    output = list(input)
    unchanged = output.copy()
    for k in range(min_fan, max_fan+1):
        for comb in itertools.combinations(unchanged, k):
            output.append(1-jnp.prod(jnp.array(comb)))
    return jnp.array(output)

for min_fan, max_fan in extra_layers:
    x_train = jax.vmap(add_second_layers, in_axes=(0, None, None))(x_train, min_fan, max_fan)
    x_test = jax.vmap(add_second_layers, in_axes=(0, None, None))(x_test, min_fan, max_fan)
    print(x_train.shape)
    print(x_test.shape)

@jax.jit
def f(x: jnp.ndarray, w: jnp.ndarray) -> float:
    # x would be all of the inputs coming in from a certain layer
    # w would be all of the weights for inputs to that layer to a given NAND gate
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w))

@jax.jit
def f_disc(x: jnp.ndarray, w: jnp.ndarray) -> int:
    return jnp.prod(jnp.where(w>0, x, 1)) 

@jax.jit
def forward(weights: jnp.ndarray, xs: jnp.ndarray) -> float:
    # the forward pass for an arbitrary neuron. 1 - the product of all the fs
    # to use vmap, I include some padding that doesn't affect the value.
    # x=1, w=0, since f(1,0)=1, so it wouldn't affect the result
    # after the product.
    return 1 - jnp.prod(jax.vmap(f)(xs, weights))

@jax.jit
def forward_disc(weights: jnp.ndarray, xs: jnp.ndarray) -> int:
    return 1 - jnp.prod(jax.vmap(f_disc)(xs, weights))

i_1 = len(arch) - 1
# i_2 = max(arch[1:])
i_3 = i_1
i_4 = max(arch)

if some_or_less == 's':
    with open("some_arrays.txt", 'r') as file:
        exec(file.read())
else:
    with open("less_arrays.txt", 'r') as file:
        exec(file.read())

epsilon = 1e-7
@jax.jit
def loss(neurons: Network, inputs: jnp.ndarray, output: jnp.ndarray) -> float:
    """
    calculates loss

    Parameters
    neurons - the network
    inputs - all of the inputs (training xs)
    output - all of the outputs (training labels or ys)
    
    Returns
    loss
    """
    pred = jax.vmap(feed_forward, in_axes=(0, None))(inputs, neurons)
    pred = jnp.clip(pred, epsilon, 1-epsilon)
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    return l1

print(x_test.shape)
print(y_test.shape)
print(loss(neurons, x_test, y_test_loss))
pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(x_test, neurons)
result = jax.vmap(image_class.evaluate)(pred, y_test)
print(jnp.sum(result)/result.size)
print(x_train.shape)
print(y_train.shape)
print(loss(neurons, x_train, y_train_loss))
pred = jax.vmap(feed_forward_disc, in_axes=(0, None))(x_train, neurons)
result = jax.vmap(image_class.evaluate)(pred, y_train)
print(jnp.sum(result)/result.size)
