import yaml
from os import getcwd, listdir
from os.path import expanduser, join, dirname, basename

import hydra
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose


class GlobalConfig:
    def set_hydra(self):
        cwd = getcwd()
        config_name = "config.yaml"
        if config_name in [p for p in listdir(cwd) if p.endswith("yaml")]:
            config_dir = cwd
        else:
            config_dir = join(expanduser("~"), ".config", "dggi_dggm")
        initialize_config_dir(version_base=None, config_dir=config_dir)
        self.default_mlf_dir = cwd
        self.default_cfg = compose(config_name=config_name)

    def update_config(self, config_path):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize_config_dir(version_base=None, config_dir=dirname(config_path))
        self.default_cfg = compose(basename(config_path))

    def save_config(
        self,
        config_path,
        training_config=None,
        generation_config=None,
        evaluation_config=None,
    ):
        if training_config is not None:
            training_config = OmegaConf.to_container(training_config)
        else:
            training_config = {}
        if generation_config is not None:
            generation_config = OmegaConf.to_container(generation_config)
        else:
            generation_config = {}
        if evaluation_config is not None:
            evaluation_config = OmegaConf.to_container(evaluation_config)
        else:
            evaluation_config = {}
        cfg = OmegaConf.to_container(self.default_cfg)
        for _cfg in [training_config, generation_config, evaluation_config]:
            for block_name, block_dict in _cfg.items():
                for param, value in block_dict.items():
                    cfg[block_name][param] = value
        cfg["hydra"] = {"run": {"dir": "."}, "output_subdir": None}
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
