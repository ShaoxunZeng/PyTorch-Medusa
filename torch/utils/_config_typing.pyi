from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

"""
This was semi-automatically generated by running

    stubgen torch.utils._config_module.py

And then manually extracting the methods of ConfigModule and converting them into top-level functions.

This file should be imported into any file that uses install_config_module like so:

    if TYPE_CHECKING:
        from torch.utils._config_typing import *  # noqa: F401, F403

    from torch.utils._config_module import install_config_module

    # adds patch, save_config, etc
    install_config_module(sys.modules[__name__])

Note that the import should happen before the call to install_config_module(), otherwise runtime errors may occur.
"""

assert TYPE_CHECKING, "Do not use at runtime"

def save_config() -> bytes: ...
def codegen_config() -> str: ...
def get_config_and_hash_with_updates(
    updates: Dict[str, Any]
) -> Tuple[Dict[str, Any], bytes]: ...
def get_hash() -> bytes: ...
def to_dict() -> Dict[str, Any]: ...
def shallow_copy_dict() -> Dict[str, Any]: ...
def load_config(config: Union[bytes, Dict[str, Any]]) -> None: ...
def get_config_copy() -> Dict[str, Any]: ...
def patch(
    arg1: Optional[Union[str, Dict[str, Any]]] = None, arg2: Any = None, **kwargs
): ...
