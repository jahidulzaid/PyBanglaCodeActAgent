"""Setup configuration for PyBanglaCodeAct."""
from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pybanglacodeact",
    version="1.0.0",
    author="Jahidul Islam, Md Ataullha, Saiful Azad",
    author_email="jahidul.cse.gub@gmail.com",
    description="A CodeAct agent for Bangla programming tasks using multilingual LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jahidulzaid/PyBanglaCodeActAgent",
    packages=find_packages(exclude=["tests", "dev_phase", "test_phase", "zero-result"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pybanglacodeact=PyBanglaCodeAct:main",
        ],
    },
    include_package_data=True,
    keywords="bangla, bengali, code-generation, llm, agent, codeact, nlp, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/jahidulzaid/PyBanglaCodeActAgent/issues",
        "Source": "https://github.com/jahidulzaid/PyBanglaCodeActAgent",
    },
)
