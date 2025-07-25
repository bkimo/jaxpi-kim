# setup.py

from setuptools import setup, find_packages
import os

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# Read the contents of your README file
# This is a robust way to do it, handling potential IOErrors
try:
    with open(os.path.join(_CURRENT_DIR, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except IOError:
    long_description = ""

setup(
    name="jaxpi",
    version="0.0.1",
    url="https://github.com/bkimo/jaxpi",
    # url="https://github.com/PredictiveIntelligenceLab/jaxpi",
    author="Modified by Bong-Sik Kim from Sifan Wang, Shyam Sankaran, Hanwen Wang",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "absl-py",
        "flax",
        "jax",
        "jaxlib",
        "matplotlib",
        "ml_collections",
        "numpy",
        "optax",
        "scipy",
        "wandb",
    ],
    extras_require={
        "testing": ["pytest"],
    },
    license="Apache 2.0",
    description="A library of PINNs models in JAX Flax.",
    long_description=long_description, # Use the variable defined above
    long_description_content_type="text/markdown",
)
