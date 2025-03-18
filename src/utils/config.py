import argparse
import os

import yaml


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            ext = ".yaml" if len(os.path.splitext(cf)) == 1 else ""
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ext)
            if not os.path.exists(cf):
                cf = os.path.basename(cf)
                repo_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )  # Adjusted for relative paths
                cf = os.path.join(
                    repo_root, "config", config_dir, cf + ext
                )  # Adjusted for relative paths
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg = dict(l, **cfg)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def parse_args_from_yaml(config_path, **kwargs):
    """Parses arguments from a YAML configuration file and command line overrides."""
    parser = argparse.ArgumentParser(description="SAR Eddy Detection Inference Demo")

    # parse config file first, then add arguments from config file
    config_path = "./config/default_config.yaml" if config_path is None else config_path
    parser.add_argument("--config", default=config_path)
    args, unknown = parser.parse_known_args()
    config = yaml_config_hook(args.config)

    # add arguments from `config` dictionary into parser, handling boolean args too
    bool_configs = [
        "multiprocessing_distributed",
        "pretrain",
        "cos",
        "mlp",
        "aug_plus",
        "use_wandb",
        "evaluate",
    ]
    for k, v in config.items():
        if k == "config":  # already added config earlier, so skip
            continue
        v = kwargs.get(k, v)
        if k in bool_configs:
            parser.add_argument(f"--{k}", default=v, type=str)
        elif k.lower() in ["seed", "pretrain_epoch_num"]:
            parser.add_argument(f"--{k}", default=v, type=int)
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    for k, v in kwargs.items():
        if k not in config:
            parser.add_argument(f"--{k}", default=v, type=type(v))

    # parse added arguments
    args, _ = parser.parse_known_args()
    for k, v in vars(args).items():
        if k in bool_configs and isinstance(v, str):
            if v.lower() in ["yes", "no", "true", "false", "none"]:
                exec(f'args.{k} = v.lower() in ["yes", "true"]')
    return args
