import jax
import jax.numpy as jnp
import optax
import random

##jax.config.update("jax_traceback_filtering", "off")

inputs = [[0,0],[0,1],[1,0],[1,1]]
inputs = [jnp.array(x) for x in inputs]
output = jnp.array([jnp.array([0,0]), jnp.array([0,1]), jnp.array([0,1]), jnp.array([1,0])])
arch = [2,1,2,2]

def f2(x, w):
    return jnp.prod(1 + jnp.multiply(x, jax.nn.sigmoid(w)) - jax.nn.sigmoid(w))
f2 = jax.jit(f2)

def f_disc(x, w):
    return jnp.prod(jnp.where(w>0, x, 1))
f_disc = jax.jit(f_disc)

## maybe later for efficiency, remove class

def flatten(weights):
    flat = []
    for layer in weights:
        for connection in layer:
            flat.append(connection)
    return flat
flatten = jax.jit(flatten)

def shape(weights):
    return [len(layer) for layer in weights]
shape = jax.jit(shape)

def update_weights(weights, new_weights):
    i = 0
    for layer, count in enumerate(shape(weights)):
        weights[layer] = jax.lax.slice_in_dim(new_weights,i,i+count)
        i += count
    return weights
##update_weights = jax.jit(update_weights)
# to "jitify" need to remove slicing

def forward(weights, xs):
    return 1 - jnp.prod(jnp.array([f2(xi,wi) for xi,wi in zip(xs, weights)]))
forward = jax.jit(forward)

def forward_disc(weights, xs):
    return 1 - jnp.prod(jnp.array([f_disc(xi,wi) for xi,wi in zip(xs, weights)]))
forward_disc = jax.jit(forward_disc)

def unpack_weights(neurons):
    weights = []
    for layer in neurons:
        for neuron in layer:
            # print(neuron)
            # print(flatten(neuron))
            weights += flatten(neuron)
    return jnp.array(weights)
unpack_weights = jax.jit(unpack_weights)

def pack_weights(weights, neurons):
    i = 0
    ws_seen = 0
    for layer in neurons:
        for neuron in layer:
            upper = ws_seen + weight_counts[i]
            i += 1
            neuron = update_weights(neuron, jnp.array(weights[ws_seen:upper]))
            ws_seen = upper
    return neurons
##pack_weights = jax.jit(pack_weights)
# to "jitify" need to remove slicing
# could pass indices as inputs, but update_weights uses jax.lax_slice_in_dim, which is incompatible with jit.

def output_weights(weights, neurons):
    output = []
    i = 0
    ws_seen = 0
    for layer in neurons:
        for neuron in layer:
            upper = ws_seen + weight_counts[i]
            i += 1
            print(weights[ws_seen:upper])
            output.append(weights[ws_seen:upper])
            ws_seen = upper
    return output

def output_circuit(neurons):
    circuits = [chr(ord('A')+i) for i in range(arch[0])]
    c2i = dict({(x,i) for i,x in enumerate(circuits)})
    indices = dict({(i,i) for i in range(arch[0])})
    empties = []
    for neuron in neurons:
        connected = set()
        for i, weight in enumerate(neuron):
            if weight > 0 and indices[i] not in empties:
                connected.add((indices[i], circuits[indices[i]]))
        connected = list(connected)
        i = len(circuits)
        if not connected:
            empties.append(i)
            indices[i] = i
            circuits.append('_')
        else:
            if len(connected) == 1:
                node = '¬' + connected[0][1]
                if len(node) > 2:
                    if node[:3] == "¬¬¬":
                        node = node[2:]
            else:
                node = '¬(' + '.'.join([element[1] for element in connected]) + ')'
            if node in c2i.keys():
                circuits.append('_')
                indices[i] = c2i[node]
            else:
                circuits.append(node)
                c2i[node] = i
                indices[i] = i
##        print(i)
##        [print(k,v) for k,v in c2i.items()]
##        print(neuron)
##        print(connected)
        print(circuits[-1])
    return circuits[-arch[-1]:]
##output_circuit = jax.jit(output_circuit)

def feed_forward(inputs, neurons):
    xs = [jnp.array(inputs)]
    for layer in neurons:
        layer_outputs = jnp.array([forward(weights, xs) for weights in layer])
        xs.append(layer_outputs)
    return xs[-1]
feed_forward = jax.jit(feed_forward)

def feed_forward_disc(inputs, neurons):
    xs = [jnp.array(inputs)]
    for layer in neurons:
        layer_outputs = jnp.array([forward_disc(weights, xs) for weights in layer])
        xs.append(layer_outputs)
    return xs[-1]
feed_forward_disc = jax.jit(feed_forward_disc)

def get_weights(layer, arch):
    global key
    weights = []
    n = sum(arch[:layer])
    mu = -2*jnp.log(n-1)
    sigma = 3
    for i in range(layer):
        weights.append(sigma * jax.random.normal(jax.random.key(key), (arch[i])) + mu)
        key += 1
    return weights

def initialise(arch):
    neurons = []
    weight_counts = []
    for i in range(1, len(arch)):
        neurons.append([])
        for j in range(arch[i]):
            weights = get_weights(i, arch)
            neurons[-1].append(weights)
            weight_counts.append(sum(shape(weights)))
    return neurons, weight_counts

epsilon = 1e-8
l2_coeff = 0.01
def jit_loss(weights, neurons):
    # calculating the output for each input.
    pred = jax.vmap(feed_forward, in_axes=(0, None))(inputs, neurons)
    pred = jnp.clip(pred, epsilon, 1-epsilon)
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    l2 = jnp.mean(1-jax.nn.sigmoid(jnp.absolute(weights)))
    return l1 + l2_coeff * l2
jit_loss = jax.jit(jit_loss)

epsilon = 1e-8
l2_coeff = 0.01
def loss(weights):
    neuronss[index] = pack_weights(weights, neuronss[index])
    pred = []
    for inp in inputs:
        act = feed_forward(inp, neuronss[index])
        act = jnp.clip(act, epsilon, 1-epsilon)
        pred.append(act)
    pred = jnp.array(pred)
    pred_logits = jnp.log(pred) - jnp.log(1-pred)
    l1 = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_logits, output))
    l2 = jnp.mean(1-jax.nn.sigmoid(jnp.absolute(weights)))
    return l1 + l2_coeff * l2
##    loss = jax.jit(loss)

def test(weights):
    neuronss[index] = pack_weights(weights, neuronss[index])
    pred = []
    for inp in inputs:
        act = feed_forward_disc(inp, neuronss[index])
        pred.append(act)
    return all([all(p==o) for p,o in zip(pred, output)])
##    test = jax.jit(test)
        
## TODO : add jit, wrap loss function in jax value and grad, vmap, jax.debug.print
## every n iterations, discretize the circuit, test and potentially stop if 100% accurate
## try on adders, other 2 bit logic gates.

print("Learning:", output, "with arch:", arch)
key = random.randint(0, 10000)

n = 5
neuronss = []
for _ in range(n):
    ne, weight_counts = initialise(arch)
    neuronss.append(ne)
solver = optax.adam(learning_rate=0.003)
paramss = [unpack_weights(neurons) for neurons in neuronss]
opt_states = [solver.init(params) for params in paramss]
cont = True
old_losses = []
for index in range(n):
    old_losses.append(round(float(loss(paramss[index])),3))
print("Losses:", old_losses)
iters = 0
while cont:
    iters += 1
    for _ in range(10):
        for index in range(n):
            grad = jax.grad(loss)(paramss[index])
            updates, opt_states[index] = solver.update(grad, opt_states[index], paramss[index])
            paramss[index] = optax.apply_updates(paramss[index], updates)
    for index, (params,neurons) in enumerate(zip(paramss, neuronss)):
        if test(params):
            print(index, "Final weights:")
            flattened = output_weights(params, neurons)
            cont = False
    if cont:
        new_losses = []
        for index in range(n):
            new_losses.append(round(float(loss(paramss[index])),3))
        for i in range(n):
            if new_losses[i] >= old_losses[i]:
                print("Restarting", i, "with new random weights, stuck in local minima", new_losses[i], old_losses[i])
                neuronss[i], _ = initialise(arch)
                paramss[i] = unpack_weights(neuronss[i])
                opt_states[i] = solver.init(paramss[i])
                new_losses[i] = round(float(loss(paramss[index])),3)
                old_losses[i] = 100
            else:
                old_losses[i] = new_losses[i]
        if iters == 10:
            print("Losses:", new_losses)
            iters = 0
print("Learnt:", output, "with arch:", arch)
print(output_circuit(flattened))
