"""Setup configuration for SignalSeeker package.

This uses the src/ layout for better package organization and isolation.
For modern Python packaging, this file is minimal; most configuration
is in pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="signalseeker",
    version="0.2.0",
    description="Educational binary classification with XGBoost for signal/background separation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Amid Nayerhoda",
    author_email="amid.nayerhoda@gmail.com",
    url="https://github.com/yourusername/SignalSeeker",
    license="MIT",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
