import jax
import jax.numpy as jnp
from typing import Tuple

# function to convert a denary number to a a jnp array of bits
def denary_to_binary_array(number: jnp.ndarray, bits: int) -> jnp.ndarray:
    return jnp.array([(jnp.right_shift(number, bits - 1 - i) & 1) for i in range(bits)], dtype=jnp.int32)

# gets parameters for learning a custom combinational logic circuit
def set_up_custom() -> Tuple[jnp.ndarray, jnp.ndarray, int, int, int]:
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