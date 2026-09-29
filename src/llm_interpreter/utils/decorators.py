import inspect
from aenum import Enum


class Category(Enum):
    def __new__(cls, name: str, help_msg: str):
        member = object.__new__(cls)
        member._value_ = name
        member.help_msg = help_msg
        return member

    # fed
    TEST = ("test", "For quick testing (similar to playground)")
    EXPERIMENT = ("experiment", "For running experiments that generate data/results")
    RESULT = ("result", "For post-processing results")
    LATEX = ("latex", "For generating LaTex tables/figures")
    DATA = ("data", "For processing dataset or collect dataset stats")


# ssalc


def subcommand(category: Category):
    def decorator(func):
        # For experiment subcommands we want shard-related knobs to be available
        # via the CLI, but not all experiment functions may explicitly declare
        # them in their signatures yet.
        original_sig = inspect.signature(func)

        def _inject_experiment_shards(sig: inspect.Signature) -> inspect.Signature:
            if category is not Category.EXPERIMENT:
                return sig

            params = list(sig.parameters.values())
            existing = {p.name for p in params}
            if "total_shards" not in existing:
                params.append(
                    inspect.Parameter(
                        "total_shards",
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=1,
                        annotation=int,
                    )
                )
            if "shard" not in existing:
                params.append(
                    inspect.Parameter(
                        "shard",
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=0,
                        annotation=int,
                    )
                )
            return sig.replace(parameters=params)

        injected_sig = _inject_experiment_shards(original_sig)

        # Wrapper needs to forward only the parameters the decorated function
        # actually accepts, so we don't break existing experiment functions.
        accepted_param_names = set(original_sig.parameters.keys())
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in original_sig.parameters.values()
        )

        def wrapper(*args, **kwargs):
            func._is_subcommand = True
            func._category = category.value
            if category is Category.EXPERIMENT and kwargs and not has_var_keyword:
                # Drop shard-related (and any other injected) kwargs if the
                # underlying function doesn't declare them. Experiment functions
                # that want sharding should declare `shard` / `total_shards` in
                # their signature and forward them into `ExperimentArgs`.
                kwargs = {
                    k: v for k, v in kwargs.items() if k in accepted_param_names
                }
            result = func(*args, **kwargs)
            post_process_subcommand(category)
            return result

        # fed
        wrapper.__annotations__ = dict(func.__annotations__)
        if category is Category.EXPERIMENT:
            wrapper.__annotations__.update({"total_shards": int, "shard": int})
        wrapper.__signature__ = injected_sig
        wrapper._is_subcommand = True
        wrapper._category = category.value
        wrapper.__doc__ = func.__doc__
        return wrapper

    # fed
    return decorator


# fed


def post_process_subcommand(category):
    match category:
        case Category.TEST:
            print("Passed!")
        case _:
            print("Done!")
    # hctam


# fed
