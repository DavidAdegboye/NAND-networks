import tensorflow as tf
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os
import numpy as np
from skimage.transform import resize
from typing import List, Tuple

Conv = Tuple[int, int, bool]

# loading the MNIST numbers dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# resizing the image from 28*28 to size*size, and from x∈[0,1] to x∈{0,1}
size = int(input("Picture size (max 28):\n"))
def preprocess_image(image: np.ndarray, s: Tuple[int, int]=(size, size), threshold: float=0.5) -> np.ndarray:
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

n = int(input("How many output neurons per number?\n"))

# turning the output label from a number to n hot encoding
@jax.jit
def preprocess_test(value: int) -> jnp.ndarray:
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
train_n = int(input("How many training samples (max 60,000)\n"))
test_n = int(input("How many testing samples (max 10,000)\n"))
x_train_resized = jnp.array([preprocess_image(img) for img in x_train[:train_n]])
x_test_resized = jnp.array([preprocess_image(img) for img in x_test[:test_n]])
y_train = jnp.array(y_train[:train_n])
y_test = jnp.array(y_test[:test_n])
y_train_new = jax.vmap(preprocess_test)(y_train)

def set_up_img() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    return x_train_resized, x_test_resized, y_train_new, y_test, train_n

# Visualising a few samples
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train_resized[i], cmap='gray')
    plt.axis('off')
    plt.title(f"Label: {y_train[i]}")
plt.show()

# similar to adder help, adding extra bits to the input which may help for the special use case of images
def add_convolution(input_image: jnp.ndarray, width: int, stride: int, min_max: bool) -> jnp.ndarray:
    """
    Returns the result of applying a min or max pooling convolution to the image with no padding

    Parameters
    input_image - the input image, a jnp array of 0s and 1s, its shape is (n,n) for some n
    width - the width of the filter
    stride - the stride of the filter
    min_max - True if we want to do max pooling, false otherwise

    Returns
    output - the result, another jnp array, its shape is also square, precisely (n-width)//stride+1
    """
    # Reshape input to [N, H, W, C] format (batch size, height, width, channels)
    # this is the format jax.lax.reduce_window wants
    input_image = input_image[jnp.newaxis, :, :, jnp.newaxis]

    # Define pooling window size and strides
    window_shape = (1, width, width, 1)
    window_strides = (1, stride, stride, 1)
    padding = 'VALID'
    if min_max:
        output = jax.lax.reduce_window(
            input_image,
            init_value=0.,
            computation=jax.lax.max,
            window_dimensions=window_shape,
            window_strides=window_strides,
            padding=padding
        )
    else:
        output = jax.lax.reduce_window(
            input_image,
            init_value=1.,
            computation=jax.lax.min,
            window_dimensions=window_shape,
            window_strides=window_strides,
            padding=padding
        )
    output_image = jnp.squeeze(output)
    return output_image

def conv_help(inputs: jnp.ndarray) -> Tuple[jnp.ndarray, int, Conv, str]:
    """
    Function to wrap taking user input, and returns a new layer. Returns the result of applying the convolution
    the user specified, the number of pixels (or elements) in this result, the width, stride and if it was min
    or max pooling, and a flag which is 'y' if they added a layer. If they didn't add a layer, the flag is 'n',
    and all the other returns are None

    Parameters
    inputs - an array of images, which themselves are square matrix of 0s and 1s

    Returns
    output - the result, another array of images, whose shapes are also square
    output.shape[1]**2 - the number of pixels in each of the output images
    (width, stride, min_max) - a tuple describing the convolution added, also enabling the same convolution to be replicated
    add_conv_help - a string which is 'y' if they added the convolution layer
    """
    add_conv_help = input("Add extra convolutional layer? Yes(y) or no(n)\n")
    if add_conv_help == 'y':
        width = int(input("What's the width of the filter?\n"))
        stride = int(input("What's the stride?\n"))
        min_max = input("Max pool(max) or min pool(min)?\n")
        output = jax.vmap(add_convolution, in_axes=(0, None, None, None))(inputs, width, stride, min_max=="max")
        return output, output.shape[1]**2, (width, stride, min_max=="max"), add_conv_help
    return None, None, None, add_conv_help

def prep_in(inputs: jnp.ndarray) -> Tuple[jnp.ndarray, List[int], List[Conv]]:
    """
    Function to prepare the input layer of the NN. Takes the array of original resized and binarized image, and adds
    as many convolution layers as the user wants. Returns a tuple of the new array, flattened, along with information
    about what convolutions were added, so they can be replicated

    Parameters
    inputs - an array of images, which themselves are square matrix of 0s and 1s

    Returns
    flattened_result - all of the layers concatenated into a flat array
    true_arch - a list describing how many pixels are in each layer
    convs - a list storing the parameters of the convolutions we did
    """
    in_list = [inputs]
    true_arch = [inputs.shape[1]**2]
    convs = []
    add_conv_help = 'y'
    while add_conv_help == 'y':
        result = conv_help(in_list[-1])
        add_conv_help = result[3]
        if add_conv_help == 'y':
            in_list.append(result[0])
            true_arch.append(result[1])
            convs.append(result[2])
    print(true_arch)
    flattened_matrices = [matrix.reshape(train_n, -1) for matrix in in_list]
    flattened_result = jnp.concatenate(flattened_matrices, axis=1)
    return flattened_result, true_arch, convs

# adds the same helper bits to the testing data that we added to the training data
def prep_test(inputs: jnp.ndarray, convs: List[Conv], samples=test_n) -> jnp.ndarray:
    """
    Function to prepare the testing data to match the training data. Returns the testing data, with the same
    extra convolutions applied that we applied to the training data.

    Parameters
    inputs - an array of images, which themselves are square matrix of 0s and 1s
    convs - a list describing the convolutions applied to training data

    Returns
    flattened_matrices - an array, containing the testing images with the convolutions applied
    """
    in_list = [inputs]
    for width, stride, min_max in convs:
        in_list.append(jax.vmap(add_convolution, in_axes=(0, None, None, None))(in_list[-1], width, stride, min_max))
    flattened_matrices = [matrix.reshape(samples, -1) for matrix in in_list]
    return jnp.concatenate(flattened_matrices, axis=1)

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

def save(extra_layers: List[Tuple[int,int]], arch: List[int], some_or_less: str, neurons: List[jnp.ndarray], convs: List[Conv], acc: str, i: int=-1) -> int:
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
        if some_or_less == 's':
            f.write(f"Using some_arrays\n")
        else:
            f.write(f"Using less_arrays\n")
        f.write(f"Size: {size}\n")
        f.write(f"Convolution layers (width, stride, min/max):\n{convs}\n")
        f.write(f"With architecture:\n{arch}\n")
        f.write(f"With extra layers:\n{extra_layers}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write("Neurons:\n")
        for layer in neurons:
            f.write(f"{jnp.sign(layer).tolist()}\n")
    return i

def load(name: str) -> Tuple[List[Tuple[int, int]], List[int], str, int, List[Conv], List[jnp.ndarray]]:
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
    if "some" in lines[0]:
        some_or_less = 's'
    else:
        some_or_less = 'l'
    s = int(lines[1].split()[-1])
    convs = eval(lines[3].strip())
    arch = eval(lines[5].strip())
    extra_layers = eval(lines[7].strip())
    neurons = []
    for line in lines[10:]:
        neurons.append(jnp.array(eval(line.strip())))
    return extra_layers, arch, some_or_less, s, convs, neurons