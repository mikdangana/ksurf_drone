from setuptools import setup, find_packages

setup(
    name="ksurf_drone",
    version="1.0.0",
    description="Attention Kalman Filter for Cloud Orchestration under highly variable workloads",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Michael Dang'ana",
    author_email="michael.dangana@mail.utoronto.ca",
    url="https://github.com/mikdangana/ksurf_drone",
    packages=find_packages(),
    install_requires=[
        "kafka-python-ng>=1.0.0",
        "jproperties>=2.0.0",
        "scipy>=1.8.0",
        "tensorflow>=2.8.0",
        "tensorflow-cpu>=2.8.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "plotly>=5.5.0",
        "PyYAML>=5.4.0",  # Added YAML support
        "scikit-learn>=0.24.0",
        "filterpy>=1.4.5"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

