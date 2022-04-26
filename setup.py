from setuptools import setup, find_packages
import os
import sys


def read(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    if sys.version < "3":
        return open(path).read()
    return open(path, encoding="utf-8").read()


README = read("README.md")
CHANGES = read("CHANGES.rst")


version = "0.1"

setup(
    name="dogsml",
    packages=find_packages(),
    namespace_packages=["dogsml"],
    version=version,
    author="Max Sitnikov",
    author_email="makc.kekc@gmail.com",
    description="ML",
    long_description="\n\n".join([README, CHANGES]),
    install_requires=[
        "numpy",
        "matplotlib",
        "h5py",
        "opencv-python",
        "tensorflow",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
