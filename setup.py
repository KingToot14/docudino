from pathlib import Path
import re
from typing import List, Tuple

from setuptools import setup, find_packages

NAME = "docudino"
DESCRIPTION = "A simple codebase that trains a Vision Transformer using the DINO framework"

URL = ""
AUTHOR = "Jacob Vanluven"
REQUIRES_PYTHON = ">=3.11"
HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

def get_requirements(path: str = HERE / "requirements.txt") -> Tuple[List[str], List[str]]:
    requirements = []
    extra_indices = []
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip("\r\n")
            if line.startswith("--extra-index-url "):
                extra_indices.append(line[18:])
                continue
            requirements.append(line)
    return requirements, extra_indices

# requirements, extra_indices = get_requirements()
version = "0.0.1"
# dev_requirements, _ = get_requirements(HERE / "requirements-dev.txt")

setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    package_data={
        "": ["*.yaml"],
    },
    # install_requires=requirements,
    # dependency_links=extra_indices,
    # extras_require={
    #     "dev": dev_requirements,
    # },
)