# In this example, we will fit a parameterised function to data. In the process of this,
# we will show how to solve optimization problems using the Optimization.jl package (https://github.com/SciML/Optimization.jl).

# We will consider the following hill function. Here we declare it as a function of its
# parameters and a single variable.
function hill_function(X, ps)
    v, K = ps
    return v * X^2 / (K^2 + X^2)
end

# Next we set a true parameter set (which we later want to recover).
v = 2.0
K = 1.2
ps_true = [v, K]

# For reference, we plot the functions.
using Plots
x_grid = 0.0:0.1:5.0
y_vals = [hill_function(x, ps_true) for x in x_grid]
plot(x_grid, y_vals; label = "True function", lw = 7)

# Next, we generate some synthetic data by sampling the true function and adding noise.
x_samples = 0.0:0.5:5.0
y_samples = [hill_function(x, ps_true) * (0.9 + 0.2 * rand()) for x in x_samples]

# For reference, we plot the sampled data (in the sample plotting plane as the true function).
plot!(x_samples, y_samples; seriestype = :scatter, label = "Sampled data", color = 1, ms = 9, alpha = 0.7)

# Now assuming that we have the data, but not the true parameter values, is it possible to recover these?
# To do this, we first define a loss function, which quantifies how good a proposed parameter set
# makes our function fit the data. Here, we will simple, for a proposed parameter set, for
# each x sample, evaluate the function and compute the squared difference to the data.
# We sum these up for all data points to get the overall loss.
function loss_function(ps_proposed, _) # Ignore the second value here, it is for compatibility with Optimization.jl.
    diff_tot = 0.0
    for i = 1:length(y_samples)
        y_proposed = hill_function(x_samples[i], ps_proposed)
        diff_tot += (y_proposed - y_samples[i])^2
    end
    return diff_tot
end

# Here, for two different proposed parameter values, we print the loss function value and
# show the fit. We note that, the parameter set that is close to the true value yields a better
# fit to the data and a smaller loss function value.
ps_proposed_1 = [1.8, 1.3]
ps_proposed_2 = [1.2, 1.8]
function plot_proposed_fit(ps_proposed)
    loss_func_eval = loss_function(ps_proposed, [])
    y_vals_proposed = [hill_function(x, ps_proposed) for x in x_grid]
    plot(x_grid, y_vals_proposed; label = "Proposed fit", lw = 7, ls = :dash, color = :blue, title = "Loss: $(round(loss_func_eval, digits=2))")
    plot!(x_samples, y_samples; seriestype = :scatter, label = "Sampled data", color = 1, ms = 9, alpha = 0.7)
end
plot(
    plot_proposed_fit(ps_proposed_1),
    plot_proposed_fit(ps_proposed_2);
    size = (1000, 400)
)

# Now, plausibly, the parameter set that minimises the loss function should yield a good fit
# to the data. Potentially, this could be an approach for finding the true parameter set.
# Here, we will use the Optimization.jl package to find which input minimises our loss function.
# The ForwardDiff package is for something called "Automatic differentiation".
# Bonus exercise: Look into this and why it is important for optimization.
using Optimization, ForwardDiff

# Now, we have to:
# (1) Wrap our loss function in an OptimizationFunction object (`AutoForwardDiff()` is again automatic differentiation-related).
opt_func = OptimizationFunction(loss_function, AutoForwardDiff())
# (2) Create an OptimizationProblem object, which bundles the function and an initial (random) guess.
ps_init = [rand(), rand()]
oprob = OptimizationProblem(opt_func, ps_init, ()) # The last argument can be ignored for our application.

# To solve our optimization problem, we need to choose an optimization algorithm. Here, we will
# use the Adam optimiser (imported from the OptimizationOptimisers package.)
using OptimizationOptimisers: Adam

# Now we solve the optimization problem using the `solve` function. The `maxiter` argument is
# how long we want to run our solve command for (higher => longer time but a better result).
sol = solve(oprob, Adam(0.1); maxiters = 2500)

# We can confirm that we got the correct parameter set. The fit is non-exact due to the noise in the data.
sol.u

# Finally, we can plot the fitted function against the true one and the data.
plot(x_grid, y_vals; label = "True function", lw = 7)
plot!(x_samples, y_samples; seriestype = :scatter, label = "Sampled data", color = 1, ms = 9, alpha = 0.7)
y_vals_fitted = [hill_function(x, sol.u) for x in x_grid]
plot!(x_grid, y_vals_fitted; label = "Fitted function", lw = 5, color = :blue, linestyle = :dash)
