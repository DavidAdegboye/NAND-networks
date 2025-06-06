import tensorflow as tf
import jax
import jax.numpy as jnp
import os
import numpy as np
from skimage.transform import resize
from typing import List, Tuple
from functools import partial
import yaml

Conv = Tuple[int, int, bool]

def set_up_img(config_dict) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    global config, size, x_train, x_test, train_n, test_n
    config = config_dict
    size = config["size"]
    n = config["n"]
    train_n = config["train_n"]
    test_n = config["test_n"]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train_resized = jnp.array([preprocess_image(img, s=(size, size)) for img in x_train[:train_n]])
    x_test_resized = jnp.array([preprocess_image(img, s=(size, size)) for img in x_test[:test_n]])
    y_train = jnp.array(y_train[:train_n])
    y_test = jnp.array(y_test[:test_n])
    y_train_new = jax.vmap(lambda x: preprocess_test(x, n=n))(y_train)
    return x_train_resized, x_test_resized, y_train_new, y_test, train_n

# resizing the image from 28*28 to size*size, and from x∈[0,1] to x∈{0,1}
def preprocess_image(image: np.ndarray, s: Tuple[int, int], threshold: float=0.5) -> jnp.ndarray:
    """
    Returns a black and white (not grayscale) image, resized to size s

    Parameters
    image - a greyscale image, in the form of an np array
    s - the target size of the black and white image, default is the size by size as inputted
    threshold - a float, the boundary we use between black and white

    Returns
    binary - an np array of shape s, with only 0s and 1s
    """
    resized = resize(image, s, anti_aliasing=True)
    binary = (resized > threshold).astype(jnp.float32)
    return binary

# turning the output label from a number to n hot encoding
@partial(jax.jit, static_argnames="n")
def preprocess_test(value: int, n: int) -> jnp.ndarray:
    """
    Returns an n hot encoded jnp array

    Parameters
    value - the "hot integer"

    Returns
    output - the n hot encoded jnp array

    Examples
    n=1
    preprocess_test(4) = [0 0 0 0 1 0 0 0 0 0]

    n=2
    preprocess_test(1) = [0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    """
    output = jnp.zeros(10 * n)
    output = jax.lax.dynamic_update_slice(output, jnp.ones(n), (value * n,))
    return output

# applying preprocessing to the data

# Visualising a few samples
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(x_train_resized[i], cmap='gray')
#     plt.axis('off')
#     plt.title(f"Label: {y_train[i]}")
# plt.show()

# @jax.jit
def get_imgs(convs: List[Tuple[int, int, int, int]]) -> List[jnp.ndarray]:
    imgs_list = tuple(jnp.expand_dims(jnp.array([preprocess_image(img, (ns,ns)) for img in x_train[:train_n]]), axis=1) for _,_,_,ns in convs)
    test_list = tuple(jnp.expand_dims(jnp.array([preprocess_image(img, (ns,ns)) for img in x_test[:test_n]]), axis=1) for _,_,_,ns in convs)
    # imgs_list = [jnp.concatenate([inputs, 1-inputs], axis=1) for inputs in imgs_list]
    # test_list = [jnp.concatenate([inputs, 1-inputs], axis=1) for inputs in test_list]
    return imgs_list, test_list

def apply_pooling(image: jnp.ndarray, pooling: Tuple[int, int, str]) -> jnp.ndarray:
    width, stride, min_max = pooling
    if min_max == "max":
        return jax.lax.reduce_window(image, 0.0, jax.lax.max, window_dimensions=(width, width), window_strides=(stride, stride), padding='VALID')
    else:
        return jax.lax.reduce_window(image, 1.0, jax.lax.min, window_dimensions=(width, width), window_strides=(stride, stride), padding='VALID')

def get_pools(pool_filters: List[Tuple[int, int, str]]) -> List[jnp.ndarray]:
    pools_list = [x_train[:train_n]] + [1-x_train[:train_n]] + [jax.vmap(apply_pooling, in_axes=(0, None))(x_train[:train_n], pool_filter) for pool_filter in pool_filters] + [jax.vmap(apply_pooling, in_axes=(0, None))(1-x_train[:train_n], pool_filter) for pool_filter in pool_filters]
    pools_test = [x_test[:test_n]] + [1-x_test[:test_n]] + [jax.vmap(apply_pooling, in_axes=(0, None))(x_test[:test_n], pool_filter) for pool_filter in pool_filters] + [jax.vmap(apply_pooling, in_axes=(0, None))(1-x_test[:test_n], pool_filter) for pool_filter in pool_filters]
    return pools_list, pools_test

# finds the most likely output based on how many neurons are "hot"
@jax.jit
def evaluate(output: jnp.ndarray, answer: int) -> bool:
    """
    Function to evaluate if the n hot encoded output matches the label

    Parameters
    output - an n hot encoded vector of 0s and 1s
    answer - an integer from 0 to 9

    Returns
    pred == answer - true iff the prediction is correct
    """
    new_output = output.reshape(10, -1)
    pred = jnp.argmax(jnp.sum(new_output, axis=1))
    return pred == answer

def save(arch: List[int], neurons_conv: List[jnp.ndarray], neurons: List[jnp.ndarray], convs: List[Conv], acc: str, i: int=-1) -> int:
    """
    Creates or overwrites a file with the information learnt by the network

    Parameters
    extra_layers - a list of the minimum and maximum fan-ins of the extra layers that may have been added by main.py
    arch - a list representing the architecture of the network
    some_or_less - a string representing if we used "some_arrays" or "less_arrays" (see some_arrays.txt and less_arrays.txt for details)
    neurons - the network learned
    convs - a list representing the convolution layers applied (these come before extra_layers)
    acc - the accuracy this network achieved
    i - default value is -1, if it's -1, it creates a new file, if it's not, we either overwrite what was in weights{i}.txt, or we make a new file called weights{i}.txt

    Returns
    i - to identify the file that we actually wrote to
    """
    if i == -1:
        i = 1
        while os.path.exists(f"weights{i}.txt"):
            i += 1
    with open(f"weights{i}.txt", "w") as f:
        f.write(f"Size: {size}\n")
        f.write(f"Convolution layers (width, stride, channels, n):\n{convs}\n")
        f.write(f"With architecture:\n{arch}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write("Convolutional layers \n")
        for layer in neurons_conv:
            f.write(f"{jnp.sign(layer).tolist()}\n")
        f.write("Neurons:\n")
        for layer in neurons:
            f.write(f"{jnp.sign(layer).tolist()}\n")
    return i

def load(name: str) -> Tuple[List[int], int, List[Tuple[int, int]], List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Loads data from a saved network

    Parameters
    name - the name of the textfile
    
    Returns
    extra_layers - a list of the minimum and maximum fan-ins of the extra layers that may have been added by main.py
    arch - a list representing the architecture of the network
    some_or_less - a string representing if we used "some_arrays" or "less_arrays" (see some_arrays.txt and less_arrays.txt for details)
    s - the size of the original resized and binarised image
    convs - a list representing the convolution layers applied (these come before extra_layers)
    neurons - the network learned
    """
    with open(name, "r") as f:
        lines = f.readlines()
    s = int(lines[0].split()[-1])
    convs = eval(lines[2].strip())
    arch = eval(lines[4].strip())
    neurons_conv = []
    switch = False
    for line in lines[7:]:
        if line.strip() == "Neurons:":
            neurons = []
            switch = True
        elif switch:
            neurons.append(jnp.array(eval(line.strip())))
        else:
            neurons_conv.append(jnp.array(eval(line.strip())))
    return arch, s, convs, neurons_conv, neurons