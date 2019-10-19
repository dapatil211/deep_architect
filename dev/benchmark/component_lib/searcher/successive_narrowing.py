import os
from six.moves import range
import numpy as np
import deep_architect.searchers.common as se
import deep_architect.utils as ut
from ..api import Searcher


# NOTE: this searcher does not do any budget adjustment and needs to be
# combined with an evaluator that does.
class SuccessiveNarrowing(Searcher):

    def __init__(self, search_space_fn, exp_config, num_initial_samples,
                 reduction_factor):
        Searcher.__init__(self, search_space_fn, exp_config)
        self.num_initial_samples = num_initial_samples
        self.reduction_factor = reduction_factor
        self.vals = [None for _ in range(num_initial_samples)]
        self.num_remaining = num_initial_samples
        self.idx = 0

        self.queue = []
        for _ in range(num_initial_samples):
            inputs, outputs = search_space_fn()
            hyperp_value_lst = se.random_specify(outputs.values())
            self.queue.append(hyperp_value_lst)

    def sample(self):
        assert self.idx < len(self.queue)
        hyperp_value_lst = self.queue[self.idx]
        (inputs, outputs) = self.search_space_fn()
        se.specify(outputs.values(), hyperp_value_lst)
        idx = self.idx
        self.idx += 1
        return inputs, outputs, hyperp_value_lst, {"idx": idx}

    def update(self, val, searcher_eval_token):
        assert self.num_remaining > 0
        idx = searcher_eval_token["idx"]
        assert self.vals[idx] is None
        self.vals[idx] = val
        self.num_remaining -= 1

        # generate the next round of architectures by keeping the best ones.
        if self.num_remaining == 0:
            num_samples = int(self.reduction_factor * len(self.queue))
            assert num_samples > 0
            top_idxs = np.argsort(self.vals)[::-1][:num_samples]
            self.queue = [self.queue[idx] for idx in top_idxs]
            self.vals = [None for _ in range(num_samples)]
            self.num_remaining = num_samples
            self.idx = 0

    def save_state(self, folderpath):
        state = {
            'num_initial_samples': self.num_initial_samples,
            'reduction_factor': self.reduction_factor,
            'vals': self.vals,
            'num_remaining': self.num_remaining,
            'idx': self.idx,
            'queue': self.queue,
        }
        ut.write_jsonfile(state, os.path.join(folderpath, 'sn_state.json'))

    def load_state(self, folderpath):
        state = ut.read_jsonfile(os.path.join(folderpath, 'sn_state.json'))
        self.num_initial_samples = state['num_initial_samples']
        self.reduction_factor = state['reduction_factor']
        self.vals = state['vals']
        self.num_remaining = state['num_remaining']
        self.idx = state['idx']
        self.queue = state['queue']


# run simple successive narrowing on a single machine.
def run_successive_narrowing(search_space_fn, num_initial_samples,
                             initial_budget, get_evaluator, extract_val_fn,
                             num_samples_reduction_factor,
                             budget_increase_factor, num_rounds,
                             get_evaluation_logger):

    num_samples = num_initial_samples
    searcher = SuccessiveNarrowing(search_space_fn, {}, num_initial_samples,
                                   num_samples_reduction_factor)
    evaluation_id = 0
    for round_idx in range(num_rounds):
        budget = initial_budget * (budget_increase_factor**round_idx)
        evaluator = get_evaluator(budget)
        for idx in range(num_samples):
            (inputs, outputs, hyperp_value_lst,
             searcher_eval_token) = searcher.sample()
            results = evaluator.eval(inputs, outputs)
            val = extract_val_fn(results)
            searcher.update(val, searcher_eval_token)
            logger = get_evaluation_logger(evaluation_id)
            logger.log_config(hyperp_value_lst, searcher_eval_token)
            logger.log_results(results)
            evaluation_id += 1

        num_samples = int(num_samples_reduction_factor * num_samples)
