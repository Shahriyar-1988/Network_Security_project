"""
The setup.py is used to define configuration of the project; such as: 
its metadata, dependencies,....
"""
from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    requirement_lst:List[str]=list()
    try:
        with open("requirements.txt") as f:
            lines=f.readlines()
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!="-e .":
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("The requirements.txt file was not found!")
    return requirement_lst

setup(
    name="Network_Security",
    version="0.0.1",
    author="Shahriyar",
    author_email="sh.abedinnejad@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()

)