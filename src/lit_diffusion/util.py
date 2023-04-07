from typing import Dict

from importlib import import_module

from lit_diffusion.constants import (
    PYTHON_CLASS_CONFIG_KEY,
    STRING_PARAMS_CONFIG_KEY,
)


def instantiate_python_class_from_string_config(
    class_config: Dict, **additional_kwargs
):
    # Assert that necessary keys are contained in config
    assert isinstance(class_config, Dict), f"{class_config} is not a dictionary."
    assert (
        PYTHON_CLASS_CONFIG_KEY in class_config.keys()
    ), f"Expected key {PYTHON_CLASS_CONFIG_KEY} but got keys: {', '.join(class_config.keys())}"

    # Recursively instantiate any further required python objects
    class_kwargs = class_config.get(STRING_PARAMS_CONFIG_KEY, dict())
    for k, v in class_kwargs.items():
        # If a parameters is a dictionary...
        if isinstance(v, Dict):
            keys = v.keys()
            # ... check if it is a valid instantiation config ...
            if (len(keys) < 2 and PYTHON_CLASS_CONFIG_KEY in keys) or (
                len(keys) < 3
                and all(
                    k in keys
                    for k in [PYTHON_CLASS_CONFIG_KEY, STRING_PARAMS_CONFIG_KEY]
                )
            ):
                # ... and if so instantiate the python object.
                class_kwargs[k] = instantiate_python_class_from_string_config(
                    class_config=v
                )

    # Get module and class names
    module_full_name: str = class_config[PYTHON_CLASS_CONFIG_KEY]
    module_sub_names = module_full_name.split(".")
    module_name = ".".join(module_sub_names[:-1])
    class_name = module_sub_names[-1]
    # Import necessary module
    module = import_module(module_name)
    # Python function call of the module attribute with specified config values
    return getattr(module, class_name)(
        **class_kwargs,
        **additional_kwargs,
    )


class TestClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b


if __name__ == "__main__":

    mock_class_config = {
        PYTHON_CLASS_CONFIG_KEY: "lit_diffusion.util.TestClass",
        STRING_PARAMS_CONFIG_KEY: {"a": 1},
    }
    print(instantiate_python_class_from_string_config(mock_class_config, b=2).__dict__)
