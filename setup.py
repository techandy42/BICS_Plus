from setuptools import setup, find_packages

setup(
    name="bics_plus",
    version="0.1.0",
    license="Apache-2.0",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "datasets==3.5.1",
        "litellm==1.67.6",
        "tenacity==9.1.2",
        "numpy==2.2.5",
        "pandas==2.2.3",
        "matplotlib==3.10.1",
        "seaborn==0.13.2",
        "tiktoken==0.9.0",
        "huggingface_hub==0.31.2",
        "pytest==8.3.4",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
    ],
)
