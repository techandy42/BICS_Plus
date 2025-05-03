from setuptools import setup, find_packages

setup(
    name="bics_plus",
    version="0.1.0",
    license="Apache-2.0",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "datasets",
        "litellm",
        "tenacity",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
    ],
)
