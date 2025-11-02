# Here, we will demonstrate how to integrate ModelingToolkitNeuralNets.jl (https://github.com/SciML/ModelingToolkitNeuralNets.jl)
# with ModelingToolkit to create hybrid ODE/Neural Network models.

# First, we declare the model components like previously.
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
@variables X(t) Y(t)
@parameters d # Note, `d` is the only non-neural network parameter here.

# Next, we use ModelingToolkitNeuralNets.jl to create a symbolic neural network representation/
using ModelingToolkitNeuralNets, Lux
nn_arch = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
)
nn, θ = SymbolicNeuralNetwork(; nn_p_name = :θ, chain = nn_arch, n_input = 1, n_output = 1)
# Here, `nn` act as a function, which takes the (symbolic) parameterisation (θ) and an input.
# Like for Lux models, both the input variable(s) and output variable(s) are in vector-form.

# We can now declare our model.
eqs_ude = [
    D(X) ~ nn([Y], θ)[1] - d*X
    D(Y) ~ X - d*Y
]
@mtkcompile xy_model_ude = System(eqs_ude, t) # Do not worry about warning.

# We can simulate the model by providing initial conditions and parameter values (for `d` and `θ`).
# `θ` is a vector variable, so have vector values. `θ` will have a "default" set of parameter values,
# which we here will use to simulate our UDE for a randomised neural network function.
using OrdinaryDiffEq, Plots
θ_vals = ModelingToolkit.getdefault(θ)
sim_cond = [X => 2.0, Y => 0.1, d => 0.5, θ => θ_vals]
tend = 45.0
oprob = ODEProblem(xy_model_ude, sim_cond, tend)
sol = solve(oprob)
plot(sol)

# If we want to see what functional form the neural network have adopted in a specific,
# `ODEProblem` we can use the following notation to evaluate the network for a single input.
input = [1.0]
oprob.ps[nn](input, oprob.ps[θ])

# And this for plotting the full functions.
x_grid = 0.0:0.1:5.0
y_vals_nn = [oprob.ps[nn]([x], oprob.ps[θ])[1] for x in x_grid]
plot(x_grid, y_vals_nn; label = "Neural Network function", lw = 5)

# Final note. If we want to simulate the model for a new condition, we can use a combination
# of the `remake` and `setp_oop` functions.
set_ps = ModelingToolkit.setp_oop(oprob, [d, θ...]) # Creates a function which allow us to generate new parameter sets.
new_d = 0.3
new_θ = [rand() for i in 1:length(θ_vals)]
ps_new = set_ps(oprob, [new_d; new_θ])
oprob_new = remake(oprob; p = ps_new)
sol_new = solve(oprob_new)
plot(sol_new)
# This approach seems a bit convoluted at first, but can improve computational performance.
