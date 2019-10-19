import argparse
import time
import threading
import logging

from deep_architect.searchers import common as se
from deep_architect import search_logging as sl
from deep_architect import utils as ut
from deep_architect.contrib.communicators.mongo_communicator import MongoCommunicator

from master import (RESULTS_TOPIC, ARCH_TOPIC, KILL_SIGNAL, PUBLISH_SIGNAL,
                    get_topic_name)
from util import create_search_space, create_evaluator

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_evaluator_fn(evaluator_config, exp_config):

    def evaluator_fn(kw_args):
        return create_evaluator(evaluator_config, exp_config, kw_args)

    return evaluator_fn


def process_config_and_args():
    parser = argparse.ArgumentParser("Worker Job for architecture search")

    parser.add_argument('--exp-config',
                        action='store',
                        dest='exp_config',
                        required=True)
    parser.add_argument('--ss-config',
                        action='store',
                        dest='ss_config',
                        required=True)
    parser.add_argument('--eval-config',
                        action='store',
                        dest='eval_config',
                        required=True)
    parser.add_argument('--search-name',
                        action='store',
                        dest='search_name',
                        required=True)
    parser.add_argument('--experiment-name',
                        action='store',
                        dest='experiment_name',
                        required=True)

    # Other arguments
    parser.add_argument('--resume',
                        '-r',
                        action='store_true',
                        dest='resume',
                        default=False)
    parser.add_argument('--mongo-host',
                        '-m',
                        action='store',
                        dest='mongo_host',
                        default='127.0.0.1')
    parser.add_argument('--mongo-port',
                        '-p',
                        action='store',
                        dest='mongo_port',
                        type=int,
                        default=27017)
    parser.add_argument('--bucket',
                        action='store',
                        dest='bucket',
                        default='deep_architect')
    parser.add_argument('--log',
                        choices=['debug', 'info', 'warning', 'error'],
                        default='info')
    parser.add_argument('--repetition', default='0')

    options = parser.parse_args()

    numeric_level = getattr(logging, options.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.log)
    logging.getLogger().setLevel(numeric_level)

    exp_config = ut.read_jsonfile(options.exp_config)

    exp_config['bucket'] = options.bucket

    exp_config['save_every'] = exp_config.get('save_every', 1)
    exp_config['search_name'] = options.search_name + '_' + options.repetition
    exp_config['search_folder'] = options.experiment_name

    evaluator_config = ut.read_jsonfile(options.eval_config)
    search_space_config = ut.read_jsonfile(options.ss_config)

    search_space_factory = create_search_space(search_space_config, exp_config)
    evaluator_fn = create_evaluator_fn(evaluator_config, exp_config)

    comm = MongoCommunicator(options.mongo_host,
                             options.mongo_port,
                             refresh_period=10)

    return comm, search_space_factory, evaluator_fn, exp_config


def save_state(comm, evaluation_id, config, state):
    logger.info('Saving state for architecture %d', evaluation_id)
    message = comm.get_value(get_topic_name(ARCH_TOPIC, config),
                             'evaluation_id', evaluation_id)
    comm.update(get_topic_name(ARCH_TOPIC, config), message, 'state', state)


def retrieve_message(message, comm, config, state):
    state['started'] = True
    data = message['data']
    if data == KILL_SIGNAL:
        logger.info('Killing worker')
        state['arch_data'] = {}
        state['specified'] = True
        comm.unsubscribe(get_topic_name(ARCH_TOPIC, config))
        comm.finish_processing(get_topic_name(ARCH_TOPIC, config),
                               message,
                               success=False)
    else:
        logger.info('Specifying architecture data %s', str(data))
        state['arch_data'] = data
        state['evaluated'] = False
        state['specified'] = True
        while not state['evaluated']:
            time.sleep(5)
        comm.finish_processing(get_topic_name(ARCH_TOPIC, config), message)


def nudge_master(comm, config, state):
    time.sleep(10)
    if not state['started']:
        comm.publish(get_topic_name(RESULTS_TOPIC, config), PUBLISH_SIGNAL)


def main():
    comm, search_space_factory, evaluator_fn, exp_config = process_config_and_args(
    )

    search_data_folder = sl.get_search_data_folderpath(
        exp_config['search_folder'], exp_config['search_name'])
    state = {
        'specified': False,
        'evaluated': False,
        'arch_data': {},
        'started': False
    }

    comm.subscribe(get_topic_name(ARCH_TOPIC, exp_config),
                   callback=lambda message: retrieve_message(
                       message, comm, exp_config, state))
    thread = threading.Thread(target=nudge_master,
                              args=(comm, exp_config, state))
    thread.start()
    step = 0
    evaluator = None
    while True:
        while not state['specified']:
            time.sleep(5)
        if state['arch_data']:
            vs = state['arch_data']['vs']
            evaluation_id = state['arch_data']['evaluation_id']
            searcher_eval_token = state['arch_data']['searcher_eval_token']
            eval_hparams = state['arch_data'].get('eval_hparams', {})
            logger.info('Evaluating architecture %d', evaluation_id)
            inputs, outputs = search_space_factory.get_search_space()
            se.specify(outputs.values(), vs)
            eval_state = comm.get_value(get_topic_name(ARCH_TOPIC, exp_config),
                                        'evaluation_id', evaluation_id)
            if eval_state is not None and 'data' in eval_state and 'state' in eval_state[
                    'data']:
                logger.info(
                    'Loading previous evaluation state for architecture %d',
                    eval_state['data']['evaluation_id'])
                eval_state = eval_state['data']['state']
            else:
                eval_state = None

            if ('reset_eval' in exp_config and
                    exp_config['reset_eval']) or evaluator is None:
                evaluator = evaluator_fn(eval_hparams)
                evaluator.pretrain()

            results = evaluator.eval(
                inputs,
                outputs,
                save_fn=lambda eval_state: save_state(comm, evaluation_id,
                                                      exp_config, eval_state),
                state=eval_state)
            logger.info('Finished evaluating architecture %d', evaluation_id)
            step += 1
            if step % exp_config['save_every'] == 0:
                logger.info('Saving evaluator state')
                evaluator.save_state(search_data_folder)

            encoded_results = {
                'results': results,
                'vs': vs,
                'evaluation_id': evaluation_id,
                'searcher_eval_token': searcher_eval_token
            }
            comm.publish(get_topic_name(RESULTS_TOPIC, exp_config),
                         encoded_results)
            state['evaluated'] = True
            state['specified'] = False
        else:
            break
    thread.join()


if __name__ == "__main__":
    main()
