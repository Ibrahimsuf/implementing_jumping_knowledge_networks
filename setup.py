from setuptools import setup, find_packages

setup(
    name="jk_networks",
    version="0.1.0",
    python_requires='>=3.8',
    install_requires=[
        "torch_geometric",
        "torch",
    ],
)
