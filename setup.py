from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

try:
    with open("README.md") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="quantumsim",
    version="0.1.0",
    author="Jack B. Jedlicki",
    author_email="jackbj@berkeley.edu",
    description="A package for quantum-inspired simulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JackBJ23/Topo-GEN",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
)
