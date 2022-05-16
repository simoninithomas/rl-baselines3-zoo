import argparse
import glob
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

from utils import ALGOS
from huggingface_hub import Repository


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="Destination Folder", type=str, default="")
    parser.add_argument("--model-id", help="Hugging Face Repo id", default="")
    parser.add_argument("--filename", help="Filename", default="")

    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=True,
                        choices=list(ALGOS.keys()))
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)

    args = parser.parse_args()

    if args.filename != "":
        cached_hf_repo_path = load_from_hub(args.model_id, args.filename)
        move_hf_repo(cached_hf_repo_path, args.folder)

    else:
        print("Copy")
        destination = os.path.join(folder, algo, f"{args.env_id}_{args.exp_id}")

        repo = Repository(destination, args.model_id)



def move_hf_repo(source, destination):
    if destination.is_dir():
        shutil.rmtree(str(destination))
    shutil.move(source, destination)
    print(f"{source} repo has been moved to {destination}")
    print(f"Now, you can continue to train your agent")


def load_from_hub(repo_id: str, filename: str) -> str:
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """
    try:
        from huggingface_hub import cached_download, hf_hub_url
    except ImportError:
        raise ImportError(
                    "You need to install huggingface_hub to use `load_from_hub`. "
                    "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    # Get the model from the Hub, download and cache the model on your local disk
    downloaded_model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        library_name="huggingface-sb3",
        library_version="2.0",
        )

    return downloaded_model_file


if __name__ == "__main__":
    main()
