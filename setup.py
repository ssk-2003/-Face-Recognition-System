"""
Setup script for Face Recognition Service
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="face-recognition-service",
    version="1.0.0",
    author="FRS Team",
    author_email="support@example.com",
    description="Production-ready Face Recognition Service",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FRS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "frs-server=main:main",
            "frs-prepare=scripts.prepare_dataset:main",
            "frs-benchmark=scripts.benchmark:main",
            "frs-evaluate=scripts.evaluate:main",
            "frs-export=scripts.export_onnx:main",
        ],
    },
)
