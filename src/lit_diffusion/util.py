import copy
from pprint import pprint
from typing import Any, Dict
from inspect import isfunction

from importlib import import_module

from lit_diffusion.constants import (
    PYTHON_CLASS_CONFIG_KEY,
    PYTHON_ARGS_CONFIG_KEY,
    PYTHON_KWARGS_CONFIG_KEY,
    INSTANTIATE_DELAY_CONFIG_KEY,
    CALL_UPON_INSTANTIATION_KEY,
)

_POSSIBLE_ARGS_CONFIG_KEYS = [
    PYTHON_ARGS_CONFIG_KEY,
    PYTHON_KWARGS_CONFIG_KEY,
    INSTANTIATE_DELAY_CONFIG_KEY,
    CALL_UPON_INSTANTIATION_KEY,
]


def instantiate_python_class_from_string_config(
    class_config: Dict,
    verbose: bool = False,
    **kwargs,
):
    class_config = copy.deepcopy(class_config)
    # Assert that necessary keys are contained in config
    assert isinstance(class_config, Dict), f"{class_config} is not a dictionary."
    assert (
        PYTHON_CLASS_CONFIG_KEY in class_config.keys()
    ), f"Expected key {PYTHON_CLASS_CONFIG_KEY} but got keys: {', '.join(class_config.keys())}"

    def recursive_call_with_check(possible_config_dict: Any):
        # If a parameters is a dictionary...
        if isinstance(possible_config_dict, Dict):
            keys = set(possible_config_dict.keys())
            # ... delay instantiation to a later call if desired ...
            if INSTANTIATE_DELAY_CONFIG_KEY in keys:
                if possible_config_dict[INSTANTIATE_DELAY_CONFIG_KEY] > 0:
                    possible_config_dict[INSTANTIATE_DELAY_CONFIG_KEY] -= 1
                    return possible_config_dict
            # ... check if it is a valid instantiation config ...
            valid_config_key_sets = [{PYTHON_CLASS_CONFIG_KEY}]
            for idx in range(len(_POSSIBLE_ARGS_CONFIG_KEYS)):
                for jdx in reversed(
                    range(idx + 1, len(_POSSIBLE_ARGS_CONFIG_KEYS) + 1)
                ):
                    valid_config_key_sets.append(
                        {PYTHON_CLASS_CONFIG_KEY, *_POSSIBLE_ARGS_CONFIG_KEYS[idx:jdx]}
                    )
            if any(keys == subset for subset in valid_config_key_sets):
                # ... and if so instantiate the python object.
                return instantiate_python_class_from_string_config(
                    class_config=possible_config_dict,
                    verbose=verbose,
                )

            # Check all levels of dict
            for config_keys, config_values in possible_config_dict.items():
                possible_config_dict[config_keys] = recursive_call_with_check(
                    config_values
                )
        # If parameters is a list
        if isinstance(possible_config_dict, list):
            # check all entries of list
            for idx, config_values in enumerate(possible_config_dict):
                possible_config_dict[idx] = recursive_call_with_check(config_values)
        return possible_config_dict

    # Recursively instantiate any further required python objects
    # ...for regular arguments
    class_args = class_config.get(PYTHON_ARGS_CONFIG_KEY, list())
    for i, v in enumerate(class_args):
        class_args[i] = recursive_call_with_check(v)
    # ...for key-word arguments
    class_kwargs = class_config.get(PYTHON_KWARGS_CONFIG_KEY, dict())
    for k, v in class_kwargs.items():
        class_kwargs[k] = recursive_call_with_check(v)

    # Get module and class names
    module_full_name: str = class_config[PYTHON_CLASS_CONFIG_KEY]
    module_sub_names = module_full_name.split(".")
    module_name = ".".join(module_sub_names[:-1])
    class_name = module_sub_names[-1]
    # Import specified module
    module = import_module(module_name)
    object_to_instantiate = getattr(module, class_name)

    # Python function call of the module attribute with specified config values
    if verbose:
        print(f"Instantiating {class_name} with the following arguments:")
        pprint(
            {"class_args": class_args, "class_kwargs": class_kwargs, "kwargs": kwargs}
        )
    # Give user the option to call it upon instantiation
    if class_config.get(CALL_UPON_INSTANTIATION_KEY, True):
        object_to_instantiate = object_to_instantiate(
            *class_args,
            **class_kwargs,
            **kwargs,
        )
    return object_to_instantiate


class TestClass:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


if __name__ == "__main__":
    mock_class_config = {
        PYTHON_CLASS_CONFIG_KEY: "lit_diffusion.util.TestClass",
        PYTHON_ARGS_CONFIG_KEY: [2],
        PYTHON_KWARGS_CONFIG_KEY: {
            "b": {
                PYTHON_CLASS_CONFIG_KEY: "lit_diffusion.util.TestClass",
                PYTHON_KWARGS_CONFIG_KEY: {
                    "a": {
                        "a": {
                            PYTHON_CLASS_CONFIG_KEY: "lit_diffusion.util.TestClass",
                            PYTHON_KWARGS_CONFIG_KEY: {"a": 1, "b": 2, "c": 3},
                        },
                        "b": 2,
                        "c": 3,
                    },
                    "b": 1,
                    "c": 3,
                },
            }
        },
    }
    pprint(mock_class_config)
    instantiate_python_class_from_string_config(mock_class_config, c=3)
