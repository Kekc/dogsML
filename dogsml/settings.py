from environs import Env
import os


__all__ = (
    "PROJECT_ROOT",
    "DATA_ROOT",
    "IMG_FOLDER",
    "DATASET_FOLDER",
    "NATURAL_IMAGES_HDF5",
    "NATURAL_IMAGES_HDF5_CONV",
    "COMPILED_MODELS_PATH",
    "TELEGRAM_BOT_TOKEN",
)


env = Env()
env.read_env()

SRC_ROOT = os.path.abspath(
    os.path.dirname(__file__)
)
PROJECT_ROOT = os.path.abspath(
    os.path.join(SRC_ROOT, "..")
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
IMG_FOLDER = "{0}/natural_images".format(DATA_ROOT)
DATASET_FOLDER = "{0}/dogs_dataset".format(DATA_ROOT)
NATURAL_IMAGES_HDF5 = "{0}/natural_images_conv.hdf5".format(
    DATASET_FOLDER
)
NATURAL_IMAGES_HDF5_CONV = "{0}/natural_images_conv.hdf5".format(
    DATASET_FOLDER
)
COMPILED_MODELS_PATH = os.path.join(SRC_ROOT, "compiled_models")

TELEGRAM_BOT_TOKEN = env.str("TELEGRAM_BOT_TOKEN")
