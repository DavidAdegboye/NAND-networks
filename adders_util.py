import jax
import jax.numpy as jnp
from typing import List, Tuple

# sets up parameters for learning an n-bit adder
bits = int(input("Input bits:\n"))
def set_up_adders() -> Tuple[jnp.ndarray, jnp.ndarray, int, int, int]:
    # if it's a n-bit adder, we need 2n inputs, n for each number 
    ins = bits*2
    num_ins = 2**ins
    inputs = jax.vmap(denary_to_binary_array)(jnp.arange(num_ins))
    output = jax.vmap(get_output)(jnp.arange(num_ins))
    outs = bits+1
    return inputs, output, ins, outs, num_ins

# function to convert a denary number to a a jnp array of bits
def denary_to_binary_array(number: jnp.ndarray, bits: int=bits*2) -> jnp.ndarray:
    return jnp.array([(jnp.right_shift(number, bits - 1 - i) & 1) for i in range(bits)], dtype=jnp.int32)

# the number it gets as input is two numbers, and so it does binary addition by doing denary addition on those numbers
def get_output(number: jnp.ndarray) -> jnp.ndarray:
    return denary_to_binary_array(number//(2**bits) + number%(2**bits), bits=bits+1)

# adding some extra bits to the inputs that may be more helpful for learning an adder
def help_adder(input: jnp.ndarray, nots: bool) -> jnp.ndarray:
    new_input = list(input)
    for i in range(bits):
        new_input.append(1-new_input[i]*new_input[i+bits])
        # so for example ABC+DEF, we're adding A NAND D, B NAND E and C NAND F
        if nots:
            new_input.append(1-new_input[i]*new_input[i+3*bits])
            new_input.append(1-new_input[i+bits]*new_input[i+2*bits])
            new_input.append(1-new_input[i+2*bits]*new_input[i+3*bits])
    return jnp.array(new_input)

def adder_help(inputs: jnp.ndarray, true_arch: List[int]) -> Tuple[jnp.ndarray, List[int], str, str|None]:
    add_adder_help = input("Add extra help for learning an adder? Yes(y) or no(n)\n")
    with_nots = None
    if add_adder_help == 'y':
        with_nots = input("Did you add a complement layer? Yes(y) or no(n)\n")
        old_ins = inputs.shape[1]
        inputs = jax.vmap(help_adder, in_axes=(0, None))(inputs, with_nots=='y')
        new_ins = inputs.shape[1]
        true_arch.append(new_ins - old_ins)
    return inputs, true_arch, add_adder_help, with_nots

# ensuring those extra bits are reflected in the output circuits function in main
def update_circuits(add_adder_help: str, circuits: List[str], with_nots: str|None, connecteds: List[List[int]]) -> Tuple[List[str], List[List[int]]]:
    if add_adder_help == 'y':
        for i in range(bits):
            circuits.append("¬(" + chr(ord('A')+i) + "." + chr(ord('A')+i+bits) + ")")
            connecteds.append([i,i+bits])
            # so for example ABC+DEF, we're adding A NAND D, B NAND E and C NAND F
            if with_nots == 'y':
                circuits.append("¬(" + chr(ord('A')+i) + ".¬" + chr(ord('A')+i+bits) + ")")
                circuits.append("¬(¬" + chr(ord('A')+i) + "." + chr(ord('A')+i+bits) + ")")
                circuits.append("¬(¬" + chr(ord('A')+i) + ".¬" + chr(ord('A')+i+bits) + ")")
                connecteds.append([i,i+3*bits])
                connecteds.append([i+bits,i+2*bits])
                connecteds.append([i+2*bits,i+3*bits])
    return circuits, connecteds