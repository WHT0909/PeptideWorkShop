from .file_io import load_data_from_csv, turn_fasta_to_csv, check_path

__all__ = [
    # file_io 模块导出功能
    'check_path', 'load_data_from_csv', 'turn_fasta_to_csv'
]

def _load_default_data_config():
    """导入配置文件"""
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'data_config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("错误：未找到配置文件 'data_config.yaml'")

DEFAULT_CONFIG = _load_default_data_config()