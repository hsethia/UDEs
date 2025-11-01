# First we import the ModelingToolkit.jl (https://github.com/SciML/ModelingToolkit.jl) package.
# This package is created to (among other things) provide a nice interface for defining equations.
using ModelingToolkit

# From the package we import the time variable (since our variables are time-dependent) and
# the differential function with respect to time.
using ModelingToolkit: t_nounits as t, D_nounits as D

# Next we declare the variables and parameters of our model.
@variables X(t) Y(t)
@parameters v K n d

# Now we can create our system of differential equations using the components we just declared.
eqs = [
    D(X) ~ v * (Y^n) / (K^n + Y^n) - d*X
    D(Y) ~ X - d*Y
]

# We store these equations in a ModelingToolkit System object, which we compile for efficiency.
@mtkcompile xy_model = System(eqs, t)

# We have now created our model. The next step is to simulate it, for which we need the
# OrdinaryDiffEq.jl (https://github.com/SciML/OrdinaryDiffEq.jl) package.
using OrdinaryDiffEq

# To simulate our model we need to declare the initial conditions and parameter values.
u0 = [X => 2.0, Y => 0.1]
ps_true = [v => 1.1, K => 2.0, n => 3.0, d => 0.5]

# We bundle the initial conditions and parameter values together into a simulation condition.
# We designate the final time point of our simulation. Then bundle everything into a `ODEProblem` object.
sim_cond = [u0; ps_true]
tend = 45.0
oprob = ODEProblem(xy_model, sim_cond, tend)

# We can now simulate our model using the `solve` function.
sol = solve(oprob)

# We can plot the simulation using the Plots.jl (https://github.com/JuliaPlots/Plots.jl) package.
using Plots
plot(sol)

# Finally, to evaluate the solution at specific time points, one can use e.g.
sol(10.0; idxs = X)
# Here we retrieve X(10.0).
