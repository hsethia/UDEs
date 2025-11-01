# We will use the Lux.jl (https://github.com/LuxDL/Lux.jl) package to create the neural
# network architecture (alternative ones, e.g. Flux.jl, exist).
using Lux

# The neural network is declared using the Chain constructor. After this, we list each layer,
# with the layer's size, activation function, and other properties.
nn_arch = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
)
# here we use the softplus activation function. A list of activation functions can be found
# here: https://lux.csail.mit.edu/stable/api/NN_Primitives/ActivationFunctions. Note that
# softplus(x) > 0, and by using this function in the last function, the overall property will
# also follow this constraint (which can be useful or problematic).

# A neural network takes a parameterisation, from which it behaves like a normal function and
# can process inputs. A random parameterisation can be created using the `setup` function.
# Here, we provide a random number generator to setup, as well as the neural network itself.
using Random
rng = Random.MersenneTwister(1234)
θ, _ = Lux.setup(rng, nn_arch)
# Note While `θ` is technically a vector, it is also a `ComponentArray`, which is a specialised
# vector with labels (hence it appears differently). However, it can also be worked like a normal vector.

# We can now apply the neural network function using the `stateless_apply` function.
input = [1.0]
LuxCore.stateless_apply(nn_arch, input, θ)
# Note 1: Lux works with vectors, so the input must be in a vector form (even in our case
# it is a single-input neural network).
# Note 2: Likewise, Lux always outputs vectors, even in the case where it is a single-output
# neural network, and the output vector thus only have a single element.
