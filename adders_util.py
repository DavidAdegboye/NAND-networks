import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict
import yaml

with open("set-up.yaml", "r") as f:
    config = yaml.safe_load(f)

# sets up parameters for learning an n-bit adder
bits = config["bits"]
def set_up_adders() -> Tuple[jnp.ndarray, jnp.ndarray, int, int, int]:
    """
   sets up a run to learn an adder

    Returns
    inputs - 2D jnp array of inputs
    output - 2D jnp array of the outputs we're trying to learn
    ins - the number of bits in the input
    outs - the number of bits in the output
    num_ins - the number of samples we have (this would be both inputs.shape[0] and output.shape[0])
    """
    # if it's a n-bit adder, we need 2n inputs, n for each number 
    ins = bits*2
    num_ins = 2**ins
    inputs = jax.vmap(denary_to_binary_array)(jnp.arange(num_ins))
    out_bits = config["out_bits"]
    output = jax.vmap(get_output)(jnp.arange(num_ins))[:,:out_bits]
    print(output.shape)
    return inputs, output, ins, out_bits, num_ins

# function to convert a denary number to a a jnp array of bits
def denary_to_binary_array(number: int, bits: int=bits*2) -> jnp.ndarray:
    """
    Converts a denary number to a jnp array of the bits

    Parameters
    number - an integer
    bits - the number of bits in the binary representation (this would also be the length of the output array)
    
    Returns
    binary list
    """
    return jnp.array([(jnp.right_shift(number, bits - 1 - i) & 1) for i in range(bits)], dtype=jnp.int32)

# the number it gets as input is two numbers, and so it does binary addition by doing denary addition on those numbers
def get_output(number: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a denary number which encodes to numbers, to the binary representation of the sum of those numbers

    Parameters
    number - an integer
    
    Returns
    binary list
    """
    return denary_to_binary_array(number//(2**bits) + number%(2**bits), bits=bits+1)

# adding some extra bits to the inputs that may be more helpful for learning an adder
def help_adder(input: jnp.ndarray, nots: bool) -> jnp.ndarray:
    """
    Returns the result of adding some extra helping bits for an adder

    Parameters
    input - a 1D jnp array, representing a single input
    nots - a boolean representing if we've added a complement helper (from add_second_layer)

    Returns
    new_input - the result, another jnp array
    """
    print(input.shape)
    new_input = list(input)
    for i in range(bits):
        new_input.append(1-new_input[i]*new_input[i+bits])
        # so for example ABC+DEF, we're adding A NAND D, B NAND E and C NAND F
        if nots:
            new_input.append(1-new_input[i]*new_input[i+3*bits])
            new_input.append(1-new_input[i+bits]*new_input[i+2*bits])
            new_input.append(1-new_input[i+2*bits]*new_input[i+3*bits])
    new_input = jnp.array(new_input)
    print(new_input.shape)
    return new_input

def adder_help(inputs: jnp.ndarray, true_arch: List[int]) -> Tuple[jnp.ndarray, List[int], str, str|None]:
    """
    Function to wrap taking user input, and adding the adder help

    Parameters
    inputs - a 2D array of inputs
    true_arch - a list representing the extra neurons added thus far

    Returns
    inputs - the updated inputs
    true_arch - the updated true_arch
    add_adder_help - a string telling us if we actually used the adder help
    with_nots - a string telling us if we used a complement layer (from add_second_layer)
    """
    add_adder_help = config["add_adder_help"]
    with_nots = None
    if add_adder_help:
        with_nots = config["with_nots"]
        old_ins = inputs.shape[1]
        inputs = jax.vmap(help_adder, in_axes=(0, None))(inputs, with_nots)
        new_ins = inputs.shape[1]
        true_arch.append(new_ins - old_ins)
    return inputs, true_arch, add_adder_help, with_nots

# ensuring those extra bits are reflected in the output circuits function in main
def update_circuits(add_adder_help: str, circuits: List[str], with_nots: str|None, connecteds: List[List[int]]) -> Tuple[List[str], List[List[int]]]:
    """
    Function to update and ensure the extra adder bits are represented when we output the circuit

    Parameters
    add_adder_help - a string telling us if we actually used the adder help
    circuits - a list of the circuits before the adder layer (so the actual inputs, and anything from add_second_layer)
    with_nots - a string telling us if we used a complement layer (from add_second_layer)
    connecteds - a list for each NAND gate, which tells us whatever before it that's connected

    Returns
    circuits - updated list of the circuits with the adder layer
    connecteds - updated list for each NAND gate, which tells us whatever before it that's connected
    """
    if add_adder_help:
        for i in range(bits):
            circuits.append("¬(" + chr(ord('A')+i) + "." + chr(ord('A')+i+bits) + ")")
            connecteds.append([i,i+bits])
            # so for example ABC+DEF, we're adding A NAND D, B NAND E and C NAND F
            if with_nots:
                circuits.append("¬(" + chr(ord('A')+i) + ".¬" + chr(ord('A')+i+bits) + ")")
                circuits.append("¬(¬" + chr(ord('A')+i) + "." + chr(ord('A')+i+bits) + ")")
                circuits.append("¬(¬" + chr(ord('A')+i) + ".¬" + chr(ord('A')+i+bits) + ")")
                connecteds.append([i,i+3*bits])
                connecteds.append([i+bits,i+2*bits])
                connecteds.append([i+2*bits,i+3*bits])
    return circuits, connecteds

def surr_trans_dict(with_nots: bool, add_help: bool) -> Dict[int, int]:
    trans_dict = dict()
    surr_bits = config["surr_bits"]
    diff = config["bits"] - surr_bits
    for i in range(surr_bits):
        trans_dict[i] = diff + i
    for i in range(surr_bits, 2*surr_bits):
        trans_dict[i] = 2 * diff + i
    if with_nots:
        for i in range(2*surr_bits, 3*surr_bits):
            trans_dict[i] = 3 * diff + i
        for i in range(3*surr_bits, 4*surr_bits):
            trans_dict[i] = 4 * diff + i
        if add_help:
            for i in range(4*surr_bits, 8*surr_bits):
                trans_dict[i] = 8 * diff + i
    elif add_help:
        for i in range(2*surr_bits, 3*surr_bits):
            trans_dict[i] = 3 * diff + i
    return trans_dict

def update_surr_arr() -> List[List[jnp.array]]:
    trans_dict = surr_trans_dict()
    out_arr = []
    for old_layer in config["surr_arr"]:
        new_layer = []
        for old_node in old_layer:
            new_node = []
            for layer_i, node_i in old_node:
                new_node.append(jnp.array([layer_i, trans_dict[node_i]]))
            new_node = jnp.stack(new_node)
            new_layer.append(new_node)
        out_arr.append(new_layer.copy())
    return out_arr
