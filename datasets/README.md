# Test datasets

This folder contains a subfolder (datasets) with all the datasets. The model's are listed in the "models" PDF-file. Each model is a UDE with one (or more) unknown functions and/or paraemters. 

For each model, attempt to recover the unknown functions and parameters from the data (this may or may not be possible). If possible, please describe the functional form/value. If it is possible, please motivate why you do not think this is the case.

### Dataset measurements
Each dataset consists of two CSV files. The first lists the initial condition values for each species. The second lists the time points of the measurements, and the values of the measured species (might be all, or a subset, of the system species). One of the model is associated with two separate datasets, please consider them individually (but in each case, using the same model).

In some cases, a dataset contains several sets of experiments. In this case, there will first be multiple sets of initial conditions in the "_initial_conditions.csv" file. Next, there will be multiple "_1_measurements_1.csv" files, where the i'th file corresponds to the measurements from the i'th initial condition.