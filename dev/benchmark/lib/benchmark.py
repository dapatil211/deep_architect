import subprocess
import os
import time
import shutil

import argparse
import progressbar
import logging

import deep_architect.utils as ut
import kube_utils
from util import create_search_space, create_searcher

logging.basicConfig()


def run_search(search_space_config,
               searcher_config,
               evaluator_config,
               exp_config,
               run_config,
               experiment_name,
               search_name,
               repetition=0):
    """
    Runs a search according to the configs provided. See the README in the
    config for additional details on the structure of the configs
    Parameters:
        - search_space_config: The config used to describe the search space.
        - searcher_config: The config used to describe the searcher.
        - evaluator_config: The config used to describe the evaluator.
        - exp_config: The config used to describe any additional details needed
                        to run the experiment. This is the config that will be
                        loaded by the master when running the experiment.
        - run_config: The config used to launch the experiment. This config will
                        not be seen by the master or workers. It is only used to
                        launch the experiment. It can be used to describe
                        environment variables that need to be set or other
                        configs needed for a distributed setting.
        - experiment_name: The name of the experiment. The log folders will be
                        structured as
                        {experiment_name}/{search_name}_{repetition}
        - search_name: The specific name of the specific search.
        - repetition: The repetition of the specific search.
    """

    # Check multiprocessing method
    config = ut.read_jsonfile(run_config)
    distribution_setting = config.get('distribution_setting', 'single_machine')
    mongo_host = config.get('mongo_host', 'localhost')
    mongo_port = config.get('mongo_port', '27017')
    bucket = config.get('bucket', '')

    if distribution_setting not in config:
        raise ValueError(
            'Need to provide a config for the specific distribtion setting')

    if distribution_setting == 'kubernetes':
        # Load the kubernetes cluster config. The output of
        # "kubectl config view" is what is loaded here
        kube_utils.load_config()

        kube_config = config['kubernetes']
        master_spec = kube_config['master']
        worker_specs = kube_config['workers']

        # Basic environment variables that can be used when designing the yaml
        # file for any kube job
        base_env_vars = {
            'EXPERIMENT_NAME': experiment_name,
            'SEARCH_NAME': search_name,
            'REPETITION': str(repetition),
            'SEARCH_SPACE_CONFIG': search_space_config,
            'SEARCHER_CONFIG': searcher_config,
            'EVALUATOR_CONFIG': evaluator_config,
            'EXP_CONFIG': exp_config,
            'MONGO_HOST': mongo_host,
            'MONGO_PORT': mongo_port,
            'BUCKET': bucket
        }

        # Create master job
        master_env_vars = master_spec.get('env_vars', {})
        master_env_vars.update(base_env_vars)
        kube_utils.create_job(master_spec['yaml_file'], master_env_vars)

        # Create worker jobs
        for worker_spec in worker_specs:
            worker_env_vars = worker_spec.get('env_vars', {})
            worker_env_vars.update(base_env_vars)
            kube_utils.create_job(worker_spec['yaml_file'], worker_env_vars)

    # If single machine, do x
    elif distribution_setting == 'single_machine':
        num_workers = config['single_machine'].get('num_workers', 1)
        subprocess.Popen([
            'python',
            'dev/benchmark/lib/master.py',
            '--exp-config',
            exp_config,
            '--ss-config',
            search_space_config,
            '--se-config',
            searcher_config,
            '--search-name',
            search_name,
            '--experiment-name',
            experiment_name,
            '--repetition',
            repetition,
            '--mongo-host',
            mongo_host,
            '--mongo-port',
            mongo_port,
            '-r',
        ])
        for worker in range(num_workers):
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(worker)
            subprocess.Popen(
                [
                    'python',
                    'dev/benchmark/lib/worker.py',
                    '--exp-config',
                    exp_config,
                    '--ss-config',
                    search_space_config,
                    '--eval-config',
                    evaluator_config,
                    '--search-name',
                    search_name,
                    '--experiment-name',
                    experiment_name,
                    '--mongo-host',
                    mongo_host,
                    '--mongo-port',
                    mongo_port,
                ],
                env=env,
            )


def benchmark_search_space(search_space_config,
                           exp_config,
                           run_config,
                           experiment_name,
                           search_name,
                           repetition=0):

    run_search(
        search_space_config,
        'dev/benchmark/configs/searcher/random.json',
        'dev/benchmark/configs/evaluator/se_opt=adam_lr=.001_epochs=25.json',
        exp_config,
        run_config,
        experiment_name,
        search_name,
        repetition=repetition)


def benchmark_searcher(searcher_config,
                       exp_config,
                       run_config,
                       experiment_name,
                       search_name,
                       repetition=0):
    run_search(
        'dev/benchmark/configs/search_space/nasnet.json',
        searcher_config,
        'dev/benchmark/configs/evaluator/se_opt=adam_lr=.001_epochs=25.json',
        exp_config,
        run_config,
        experiment_name,
        search_name,
        repetition=repetition)


def create_arch_configs(arch_configs_file,
                        searcher_config,
                        search_space_config,
                        num_configs=128):
    ssf = create_search_space(search_space_config, {})
    searcher = create_searcher(searcher_config, {}, ssf)

    configs = []

    logging.info('Generating architecture configs')

    for _ in progressbar.progressbar(range(num_configs)):
        _, _, vs, _ = searcher.sample()
        configs.append(vs)
    ut.write_jsonfile(configs, arch_configs_file)


def benchmark_evaluator(evaluator_config,
                        exp_config,
                        run_config,
                        experiment_name,
                        search_name,
                        repetition=0):
    searcher_config = "dev/benchmark/configs/searcher/fixed.json"
    config = ut.read_jsonfile(searcher_config)
    arch_configs_file = config['kw_args']['filename']
    local_filepath = os.path.join(experiment_name, arch_configs_file)
    exp_config_dict = ut.read_jsonfile(exp_config)

    # If architecture config file does not exist, create one locally under
    # the experiment folder
    if not ut.file_exists(local_filepath):
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        create_arch_configs(local_filepath,
                            'dev/benchmark/configs/searcher/random.json',
                            'dev/benchmark/configs/search_space/nasnet.json',
                            num_configs=exp_config_dict.get(
                                'num_evaluations', 128))

    # Launch jobs
    run_search('dev/benchmark/configs/search_space/nasnet.json',
               searcher_config,
               evaluator_config,
               exp_config,
               run_config,
               experiment_name,
               search_name,
               repetition=repetition)

    # If running experiment through kubernetes, need to copy architecture config
    # file to the master pod for the search (aka where the searcher will run)
    run_config = ut.read_jsonfile(run_config)
    if run_config['distribution_setting'] == 'kubernetes':
        # Loop while the file hasn't been copied. This is probably because the
        # pod is still being created
        while True:
            try:
                # Get the details of the master pod
                master_pod = kube_utils.get_pod_with_name(
                    [search_name, experiment_name,
                     str(repetition), 'master'])

                # Copy to the architecture config file to the project root
                # directory
                kube_utils.copy_to_pod(
                    master_pod, local_filepath,
                    os.path.join(run_config['kubernetes'].get('root_dir', '/'),
                                 arch_configs_file))
                break
            except:
                time.sleep(2.0)
    elif run_config['distribution_setting'] == 'single_machine':
        shutil.copyfile(local_filepath, arch_configs_file)


def main():
    parser = argparse.ArgumentParser("Launch benchmark jobs")
    parser.add_argument('--benchmark',
                        '-b',
                        choices=['search_space', 'searcher', 'evaluator'],
                        action='store',
                        dest='benchmark_type',
                        default='search_space',
                        help='The type of component to be benchmarked')
    parser.add_argument('--run-config',
                        '-r',
                        required=True,
                        action='store',
                        dest='run_config')
    parser.add_argument('--exp-config',
                        '-e',
                        required=True,
                        action='store',
                        dest='exp_config')
    parser.add_argument('--experiment-name',
                        '-n',
                        required=True,
                        action='store',
                        dest='experiment_name')
    parser.add_argument('--search-name',
                        '-s',
                        required=True,
                        action='store',
                        dest='search_name')
    parser.add_argument('--repetition',
                        default='0',
                        action='store',
                        dest='repetition')
    parser.add_argument(
        '--ss-config',
        action='store',
        dest='search_space_config',
        help=
        'Location of config defining search space component to be benchmarked')
    parser.add_argument(
        '--se-config',
        action='store',
        dest='searcher_config',
        help='Location of config defining searcher component to be benchmarked')
    parser.add_argument(
        '--eval-config',
        action='store',
        dest='evaluator_config',
        help='Location of config defining evaluator component to be benchmarked'
    )

    options = parser.parse_args()

    if (options.benchmark_type == 'search_space' and
            options.search_space_config != ''):
        benchmark_search_space(options.search_space_config, options.exp_config,
                               options.run_config, options.experiment_name,
                               options.search_name, options.repetition)
    elif (options.benchmark_type == 'searcher' and
          options.searcher_config != ''):
        benchmark_searcher(options.searcher_config, options.exp_config,
                           options.run_config, options.experiment_name,
                           options.search_name, options.repetition)
    elif (options.benchmark_type == 'evaluator' and
          options.evaluator_config != ''):
        benchmark_evaluator(options.evaluator_config, options.exp_config,
                            options.run_config, options.experiment_name,
                            options.search_name, options.repetition)
    else:
        raise ValueError('Proper arguments not provided')


if __name__ == '__main__':
    main()
