from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirement()->List[str]:
    """
    This function returns a list of requirements
    """
    requirement_list:List[str] = []
    try: 
        with open('requirements.txt', 'r') as file:
            # read lines from the file_obj
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                # ignore empty lines and -e .
                if requirement and requirement != "-e .":
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
        
    return requirement_list  

print(get_requirement())
    
setup(
    name="OilGasMarketOptimisation",
    version="0.0.1",
    author="Sookchand Harripersad",
    author_email="sookchand38@gmail.com",
    packages=find_packages(),
    install_requires=get_requirement("requirements.txt")
)
