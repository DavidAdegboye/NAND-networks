# NAND-networks
My Part II Project
TODO:
from the lowest level, can include padding on xs and weights, to make everything a jnp array and completely utilise jax efficiencies.
this would become a part of the forward function, allowing me to use jax.vmap instead of list comprehension

if I can jax.jit pack_weights, would get much better performance, since this would allow me to jax.jit loss

want to add sigma as an argument for the initialisation function. I found a hand wavy optimal initialisation given sigma = 3.
In practice all that matters is that the initialisation is good enough. But it seems not to be. This is causing me to question
my assumptions. One of these assumptions is sigma = 3, which was chosen arbitrarily. By allowing sigma to be a parameter, I can more
easily try different initialisations, and even try some different initialisations in parallel.
One of such assumptions is even that we should use a normal. This assumption was mainly because it made things easier to model.

make layers wider, scale up
potentially include other gates for better optimisation
4+4 bit adder, vary intialisation, gates, layer sizes
measure loss curves, and training times.
lower priority: different loss functions and optimisers
