#!/usr/bin/env python3
"""
Setup script for QuantElite - Elite Quantitative Trading Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantelite",
    version="1.0.0",
    author="PArth Panchal",
    author_email="parthpanchal@Mac.lan",
    description="QuantElite - Elite Quantitative Trading Platform with Advanced Technical Analysis and Quantitative Trading Strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parthpanchal/quantelite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "quantelite=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*.html", "static/*"],
    },
)
