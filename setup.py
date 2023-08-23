from setuptools import setup, find_namespace_packages
import os

pkg_data_directories = []

def get_package_data(pkg_name, data_directories = pkg_data_directories):
    pkg_data = {}
    for dir in data_directories:
        for t in os.walk(dir):
            subdir = t[0].replace(os.path.sep, ".")
            pkg_data[f"{pkg_name}.{subdir}"] = ['*']
    return pkg_data

def read_requirements_from_file(requirement_file="requirements.txt"):
    requirements = []
    with open(requirement_file) as fd:
        for req in fd:
            requirements.append(req)
            
    return requirements

setup(
    name="XAI",
    version="0.1",
    package_dir={"": "src"},
    package_data=get_package_data("XAI"),
    packages=find_namespace_packages(where="src"),
    include_package_data=True,
    
    install_requires=read_requirements_from_file(),
    
    entry_points={
        "console_scripts": [
            "XAI=XAI.__main__:main"
        ]
    },
)