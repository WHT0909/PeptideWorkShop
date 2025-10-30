from .KL_diffEntropy import get_KL_diffEntropy

__all__ = [
    'get_KL_diffEntropy'
]

def _load_default_data_config():
    """导入配置文件"""
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'feature_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("错误：未找到配置文件 'feature_config.yaml'")

DEFAULT_CONFIG = _load_default_data_config()