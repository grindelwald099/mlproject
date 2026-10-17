from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """This function will return the list of requirements"""
    requirements=[]
    with open(file_path) as f:
        requirements = f.read().splitlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
        return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Veeru',
    author_email='veeru.astro@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")[:-1]
)
    
    
