__version__ = "0.1.0"

__all__ = [
    "ContractError",
    "get_contract",
    "ObservationBuilder",
    "OnnxPolicy",
    "PolicyRunner",
]


def __getattr__(name):
    if name in ("ContractError", "get_contract"):
        from . import contract

        return getattr(contract, name)
    if name == "ObservationBuilder":
        from .observation import ObservationBuilder

        return ObservationBuilder
    if name == "OnnxPolicy":
        from .policy import OnnxPolicy

        return OnnxPolicy
    if name == "PolicyRunner":
        from .runner import PolicyRunner

        return PolicyRunner
    raise AttributeError(name)
