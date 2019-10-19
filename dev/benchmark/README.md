# Benchmarking with DeepArchitect
## Usage
The launch file for the benchmark is [benchmark.py](lib/benchmark.py).

```
usage: Launch benchmark jobs [-h]
                             [--benchmark {search_space,searcher,evaluator}]
                             --run-config RUN_CONFIG --exp-config EXP_CONFIG
                             --experiment-name EXPERIMENT_NAME --search-name
                             SEARCH_NAME [--repetition REPETITION]
                             [--ss-config SEARCH_SPACE_CONFIG]
                             [--se-config SEARCHER_CONFIG]
                             [--eval-config EVALUATOR_CONFIG]
```

To run this script, the type of component being benchmarked (search space, searcher, evaluator) needs to be passed in, along with the locations of the five configuration files, and the experiment and search names.

The five configuration files configure different parts of the NAS experiment.
* The run configuration is used to decide where/how the experiment should be launched (e.g. single machine vs distributed with kubernetes, google cloud bucket where data should be stored, the number of workers).
* The experiment configuration is used to provide information about the overall experiment (e.g. how many evaluations the experiment should go through).
* The search space configuration, searcher configuration, and evaluator configuration are used (along with the general experiment configuration) to create the search space, searcher, and evaluator respectively.

Examples of each of these types of configurations can be found in [configs](configs/).

The experiment and search names are used to group together log files that are generated. An experiment consists of several searches. A search folder is created for every run of the benchmark.

## Supported Settings
We currently support multi-GPU single machine and Kubernetes (specifically on GKE). The specific setting used is configured by the run configuration. To see an example of a single machine configuration, see [single_machine.json](configs/run/single_machine.json).

To see an example of kubernetes configuration, see [kube_25_tpus.json](configs/run/kube_25_tpus.json). With this configuration running distributed architecture search on TPUs is simple:
1. Create the cluster with the relevant node selectors and TPU support enabled.
2. Run `gcloud container clusters get-credentials [CLUSTER_NAME]` once the cluster is provisioned.
3. Set the desired number of workers of each type in [kube_25_tpus.json](configs/run/kube_25_tpus.json).
4. Launch!