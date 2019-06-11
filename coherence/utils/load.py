import yaml
from allennlp.common import Params
from allennlp.common.params import parse_overrides, with_fallback


def load_params(param_file, overrides):
    """Param loader with YAML support."""
    if not param_file.endswith(('.yaml', '.yml')):
        return Params.from_file(param_file, overrides)

    with open(param_file) as f:
        file_dict = yaml.safe_load(f)

    overrides_dict = parse_overrides(overrides)
    param_dict = with_fallback(preferred=overrides_dict, fallback=file_dict)
    return Params(param_dict)
