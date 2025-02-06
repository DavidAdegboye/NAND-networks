import jax
import jax.numpy as jnp
from typing import Tuple

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
def set_up_custom() -> Tuple[jnp.ndarray, jnp.ndarray, int, int, int]:
    """
    Takes some user input and sets up a run to learn a custom combinational logic circuit

    Returns
    inputs - 2D jnp array of inputs
    output - 2D jnp array of the outputs we're trying to learn
    ins - the number of bits in the input
    outs - the number of bits in the output
    2**ins - the number of samples we have (this would be both inputs.shape[0] and output.shape[0])
    """
    # number of input bits
    ins = int(input("How many inputs?\n"))
    # number of output bits
    outs = int(input("How many outputs?\n"))
    output_list = []
    for i in range(outs):
        next_out = input(f"Enter the desired output for the neuron {i+1}, separated by spaces. (There should be {2**ins})\n")
        output_list.append([int(x) for x in next_out.split()])
    output = jnp.transpose(jnp.array(output_list))
    inputs = jax.vmap(denary_to_binary_array, in_axes=(0, None))(jnp.arange(2**(ins)), ins)
    return inputs, output, ins, outs, 2**ins