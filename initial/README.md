# Sample code - Hybrid neural network/ordinary differential equation project
This repository contain some sample code for various (Julia) workflows that could be useful for the project.

### Julia information
Julia (https://julialang.org/) is a recently introduced scripting language in the same vein as Python Matlab, and R.
- Generally, it is recommended to download, install, and update it using "juliaup": https://julialang.org/install/
- Visual Studio Code (https://code.visualstudio.com/docs/languages/julia) have a Julia extension and is strongly recommended as the IDe of choice.
- GitHub's Copilot (https://code.visualstudio.com/docs/copilot/overview) can be useful. Also, e.g. ChatGPT can be useful for questions.
- A full Julia documentation can be found [here](https://docs.julialang.org/en/v1/). There are multiple resources for learning about the language: https://julialang.org/learning/.

### Repository initialisation  
The code in this repository depends on a number of packages (a full list can be found in the "Project.toml" file). The download them you can run the following Julia commands:
```julia
using Pkg           # Activating the Julia package manager.
Pkg.activate(".")   # Activate the local Julia environment.
Pkg.instantiate()   # Download all the packages of the environment.
```

### Repository structure
The repository contains the following demos:
- "ODE_sim_demo.jl": How to declare and simulate an ODE model in Julia.
- "nn_evaluation_demo.jl": How to create an evaluate a neural network in Julia.
- "function_fitting_demo.jl: How to fit the parameters of a function to data through loss function optimisation.
- "ude_creation_demo.jl": How to declare a hybrid ODE/Neural network model (also called a *universal differential equation*).

It is also recommended to, when necessary, check the documentation of relevant packages. E.g. ChatGPT should also be able to provide advice. Finally, it is possible that ChatGPT should be able to convert these tutorials to e.g. Python code, in case you prefer a non-Julia language.

