from .add_gaussian_noise import add_gaussian_noise

__all__ = [
    'add_gaussian_noise'
]

def _load_default_data_config():
    """导入配置文件"""
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'augmentation_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("错误：未找到配置文件 'augmentation_config.yaml'")

DEFAULT_CONFIG = _load_default_data_config()