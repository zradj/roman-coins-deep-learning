from pathlib import Path


def get_coin_config():
    return {
        "in_channels": 3,
        "image_size": 256,
        "num_classes": 100,
        "channel_dim": 256,
        "token_dim": 128,
        "depth": 6,
        "patch_size": 8,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "num_epochs": 20,
        "model_folder": "weights_coins",
        "model_basename": "kan_mixer_coins_",
        "resume_epoch": None,
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
