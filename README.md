## TSMF: Network Latency Estimation using Matrix Factorization and Time Series Forecasting

This repository includes all experiments and models used in the aforementioned paper. All modules that were used in the experiments are under `main.codebase.evaluator` and all models under `main.codebase.models`

All *configuration* parameters used for the models in the experiments are in `main/codebase/config.py`.

This repository uses `Python=3.7`. A virtual environment has to be created using a tool of your choice and then all necessary modules installed using:

`pip install -r requirements.txt`

Or the environment can be installed using Conda: [Creating an environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

## Running reproducibility experiments

Once the virtual environment has been created and activated the accuracy experiments can be reproduced as follows:

`python -m main.codebase.evaluator.evaluator_driver [test_size] [missing_ratio] [fpath] [-a all_models | -m models] [-p processes] [-v verbose]`

`test_size` is the number of matrices that models will be evaluated on </br>
`missing_ratio` is the ratio of missing entries that the matrices will have </br>
`fpath` is the directory path which must lead to a directory with distinct matrices</br>
`-a | -m` when using `-a` all models included in `evaluatory_driver.py` will be used. If using `-m`, then the flag has to be followed by the names of the models


*Optional Parameters*</br>
`-p` the number of separate processes that will be launched for faster evaluation when using large number of models and large `test_size`</br>
`-v` controls the verbosity level

Example:
> python -m main.codebase.evaluator.evaluator_driver 10 0.7 "Data/Seattle/" -m  TSMF SimpleMF -p 5 -v

THe evaluator will then generate two files under directories `output/logs` and `output/Accuracy`. A `csv` file will be generated as `output/Accuracy/evaluation_run_{timestamp}-{test_size}-{missing_ratio}-{fpath}.csv` that includes all estimations made by the models along with `matrix-id` information and the *true* values. Under `output/logs` a log file will be generated that includes all verbosity statements and the parameter values used when generating the models. 
