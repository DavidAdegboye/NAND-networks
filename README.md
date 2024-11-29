# NAND-networks
My Part II Project
TODO:

want to add sigma as an argument for the initialisation function. I found a hand wavy optimal initialisation given sigma = 3.
In practice all that matters is that the initialisation is good enough. But it seems not to be. This is causing me to question
my assumptions. One of these assumptions is sigma = 3, which was chosen arbitrarily. By allowing sigma to be a parameter, I can more
easily try different initialisations, and even try some different initialisations in parallel.
One of such assumptions is even that we should use a normal. This assumption was mainly because it made things easier to model.

make layers wider, scale up
potentially include other gates for better optimisation
4+4 bit adder, vary intialisation, gates, layer sizes
measure loss curves, and training times.

include logging of losses and training times
lower priority: different loss functions and optimisers
also how they vary with adder size
test different shapes - flat, increasing, decreasing
if I find that it's agnostic to certain hyperparameters, stop worrying
sanity check the maths

embedding learnt circuits into bigger domains
optax chaining gradient transforms, masking, can optimise different parts of the network differently - could have a lower learning rate for original connections, higher for new ones.
temporal l2 coefficient increasing.

value consistency - don't stop when you find a solution, see how consistent certain parameters are at getting solutions.
meeting with Rob - slides/presentation
