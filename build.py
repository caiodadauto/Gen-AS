from shutil import copyfile
from os import makedirs
from os.path import abspath, dirname, join, expanduser


def build(setup_kwargs):
    base_config_file = join(dirname(abspath(__file__)), "config.yaml")
    target_dir = join(expanduser("~"), ".config", "digg_dggm")
    makedirs(target_dir, exist_ok=True)
    copyfile(base_config_file, join(target_dir, "config.yaml"))
