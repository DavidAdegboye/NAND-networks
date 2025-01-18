import tensorflow as tf
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from skimage.transform import resize
from typing import List, Tuple

# loading the MNIST numbers dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# resizing the image from 28*28 to size*size, and from x∈[0,1] to x∈{0,1}
size = int(input("Picture size (max 28):\n"))
def preprocess_image(image, size=(size, size), threshold=0.5):
    resized = resize(image, size, anti_aliasing=True)
    binary = (resized > threshold).astype(jnp.float32)
    return binary

n = int(input("How many output neurons per number?\n"))

# turning the output label from a number to n hot encoding
def preprocess_test(value: int) -> jnp.ndarray:
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

def set_up_img():
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
    # Reshape input to [N, H, W, C] format (batch size, height, width, channels)
    input_image = input_image[jnp.newaxis, :, :, jnp.newaxis]

    # Define pooling window size and strides
    window_shape = (1, width, width, 1)
    window_strides = (1, stride, stride, 1)
    padding = 'VALID'  # No padding
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

def conv_help(inputs: jnp.ndarray) -> Tuple[jnp.ndarray, int, Tuple[int, int, bool], str]:
    add_conv_help = input("Add extra convolutional layer? Yes(y) or no(n)\n")
    if add_conv_help == 'y':
        width = int(input("What's the width of the filter?\n"))
        stride = int(input("What's the stride?\n"))
        min_max = input("Max pool(max) or min pool(min)?\n")
        output = jax.vmap(add_convolution, in_axes=(0, None, None, None))(inputs, width, stride, min_max=="max")
        return output, output.shape[1]**2, (width, stride, min_max=="max"), add_conv_help
    return None, None, None, add_conv_help

def prep_in(inputs: jnp.ndarray) -> Tuple[jnp.ndarray, List[int], List[Tuple[int, int, bool]]]:
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
def prep_test(inputs: jnp.ndarray, convs: List[Tuple[int, int, bool]]) -> jnp.ndarray:
    in_list = [inputs]
    for width, stride, min_max in convs:
        in_list.append(jax.vmap(add_convolution, in_axes=(0, None, None, None))(in_list[-1], width, stride, min_max))
    flattened_matrices = [matrix.reshape(test_n, -1) for matrix in in_list]
    return jnp.concatenate(flattened_matrices, axis=1)

# finds the most likely output based on how many neurons are "hot"
def eval(output: jnp.ndarray, answer: int) -> jnp.ndarray:
    new_output = output.reshape(10, -1)
    pred = jnp.argmax(jnp.sum(new_output, axis=1))
    return pred == answer
