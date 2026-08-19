import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from hacod.utils.seed import set_seed

from hacod.datasets.label_mapping import (
    load_dataset_mappings,
)

from hacod.datasets.dataset_builder import (
    build_train_dataset,
    copy_val_dataset,
    write_yaml,
)

from hacod.continual.replay_memory import (
    rebuild_memory_from_all_tasks,
)

from hacod.losses.hacod_loss import (
    attach_hacod_loss,
)

from hacod.utils.evaluation import (
    evaluate_on_seen_tasks,
)

def load_config(config_path):

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)


def main(args):

    config = load_config(
        args.config
    )

    seed = config[
        "experiment"
    ]["seed"]

    set_seed(seed)

    # load mappings
    mappings = (
        load_dataset_mappings()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--start-task",
        type=int,
        default=1
    )

    args = parser.parse_args()

    main(args)