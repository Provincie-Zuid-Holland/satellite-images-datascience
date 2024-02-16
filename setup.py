from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="nso_ds_classes",  # Replace with your own username
    version="1.0.0",
    author="Michael de Winter",
    author_email="m.r.dewinter88@live.nl",
    description="Satellite object recognition models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Provincie-Zuid-Holland/satellite-images-nso-datascience",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "earthpy==0.9.4",
        "geopandas==0.14.2",
        "scikit-learn==1.4.0",
        "rasterio==1.3.9",
        "fiona==1.9.5",
        "pyarrow==15.0.0",
        "fastparquet==2024.2.0",
    ],
)
