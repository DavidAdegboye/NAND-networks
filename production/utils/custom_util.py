import jax
import jax.numpy as jnp
from typing import Tuple
import yaml

# function to convert a denary number to a a jnp array of bits
def denary_to_binary_array(number: int, bits: int) -> jnp.ndarray:
    """
    Converts a denary number to a jnp array of the bits

    Parameters
    number - an integer
    bits - the number of bits in the binary representation (this would also be the length of the output array)
    
    Returns
    binary list
    """
    return jnp.array([(jnp.right_shift(number, bits - 1 - i) & 1) for i in range(bits)], dtype=jnp.int32)

# gets parameters for learning a custom combinational logic circuit
def set_up_custom(config_dict) -> Tuple[jnp.ndarray, jnp.ndarray, int, int, int]:
    """
    Takes some user input and sets up a run to learn a custom combinational logic circuit

    Returns
    inputs - 2D jnp array of inputs
    output - 2D jnp array of the outputs we're trying to learn
    ins - the number of bits in the input
    outs - the number of bits in the output
    2**ins - the number of samples we have (this would be both inputs.shape[0] and output.shape[0])
    """
    global config
    config = config_dict
    # number of input bits
    ins = config["ins"]
    # number of output bits
    outs = config["outs"]
    output = jnp.array(config["output"])
    inputs = jax.vmap(denary_to_binary_array, in_axes=(0, None))(jnp.arange(2**(ins)), ins)
    return inputs, output, ins, outs, 2**ins