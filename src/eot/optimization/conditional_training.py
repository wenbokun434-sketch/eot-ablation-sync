"""Reuse the exact O07 training function with an isolated O12 loss factory.

FunctionType binds a private copy of its globals. No original module globals or
source are changed, and no training-loop/optimizer/mask code is copied. The
binding and its loss factory are included in every new run's source hash.
"""
from types import FunctionType
from . import training
from .conditional_config import gate, OUTPUT
from .config import canonical_json
from .focal import build_loss
from .runs import RunIdentity, source_fingerprint
from .splits import digest

SOURCES = (*training.TRAINING_SOURCES, "src/eot/optimization/focal.py",
           "src/eot/optimization/conditional_config.py",
           "src/eot/optimization/conditional_training.py",
           "src/eot/optimization/conditional.py")


def prepare(root, f, j, candidate, seed):
    config, trigger = gate(root, f, candidate)
    a = training.load_optimization_config(root / "configs/optimization/A.yaml")
    # O07 verifies the unchanged inputs, G/E/H split, statistics and constants.
    reference = training.prepare_inner_run(a, stage="screening" if seed == 42 else "stability",
                                          outer_fold=f, inner_fold=j, seed=seed)
    metadata = {**reference.identity.metadata, "stage": "conditional", "candidate_id": candidate,
                "config_sha256": config.config_hash, "code_sha256": source_fingerprint(root, SOURCES),
                "trigger_sha256": digest(trigger)}
    run_id = digest(metadata)
    identity = RunIdentity(root, run_id, canonical_json(metadata),
        f"{OUTPUT}/outer_{f}/inner_{j}/{candidate}/seed_{seed}/{run_id}")
    return training.InnerTrainingPlan(config, reference.context, identity, reference._inputs_json)


def fit(plan, store, *, device_name, resume=False, stop_after_epoch=None):
    namespace = dict(training._fit_inner.__globals__)
    namespace["build_loss"] = build_loss
    bound = FunctionType(training._fit_inner.__code__, namespace,
                         "conditional_fit_inner", training._fit_inner.__defaults__,
                         training._fit_inner.__closure__)
    bound.__kwdefaults__ = training._fit_inner.__kwdefaults__
    return bound(plan, store, device_name=device_name, resume=resume,
                 stop_after_epoch=stop_after_epoch)
