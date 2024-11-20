# NAND-networks
My Part II Project
TODO:
from the lowest level, can include padding on xs and weights, to make everything a jnp array and completely utilise jax efficiencies.
this would become a part of the forward function, allowing me to use jax.vmap instead of list comprehension

if I can jax.jit pack_weights, would get much better performance, since this would allow me to jax.jit loss
