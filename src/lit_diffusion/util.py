from typing import Dict

from importlib import import_module

from lit_diffusion.constants import (
    PYTHON_CLASS_CONFIG_KEY,
    PYTHON_ARGS_CONFIG_KEY,
    PYTHON_KWARGS_CONFIG_KEY,
)

_POSSIBLE_CONFIG_KEYS = [
    PYTHON_CLASS_CONFIG_KEY,
    PYTHON_ARGS_CONFIG_KEY,
    PYTHON_KWARGS_CONFIG_KEY,
]


def instantiate_python_class_from_string_config(
    class_config: Dict, **kwargs,
):
    # Assert that necessary keys are contained in config
    assert isinstance(class_config, Dict), f"{class_config} is not a dictionary."
    assert (
        PYTHON_CLASS_CONFIG_KEY in class_config.keys()
    ), f"Expected key {PYTHON_CLASS_CONFIG_KEY} but got keys: {', '.join(class_config.keys())}"

    def recursive_call_with_check(possible_config_dict):
        # If a parameters is a dictionary...
        if isinstance(possible_config_dict, Dict):
            keys = set(possible_config_dict.keys())
            # ... check if it is a valid instantiation config ...
            if any(
                keys == subset
                for subset in [
                    set(_POSSIBLE_CONFIG_KEYS[:idx])
                    for idx in range(1, len(_POSSIBLE_CONFIG_KEYS) + 1)
                ]
            ):
                # ... and if so instantiate the python object.
                return instantiate_python_class_from_string_config(
                    class_config=possible_config_dict
                )

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
    # Python function call of the module attribute with specified config values
    return getattr(module, class_name)(
        *class_args,
        **class_kwargs,
        **kwargs,
    )


class TestClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b


if __name__ == "__main__":

    mock_class_config = {
        PYTHON_CLASS_CONFIG_KEY: "lit_diffusion.util.TestClass",
        PYTHON_ARGS_CONFIG_KEY: [
            {
                PYTHON_CLASS_CONFIG_KEY: "lit_diffusion.util.TestClass",
                PYTHON_ARGS_CONFIG_KEY: [2],
                PYTHON_KWARGS_CONFIG_KEY: {"b": 1},
            }
        ],
        PYTHON_KWARGS_CONFIG_KEY: {"b": 1},
    }
    print(
        instantiate_python_class_from_string_config(
            mock_class_config,
        ).__dict__
    )
