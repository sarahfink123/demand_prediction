from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name='demand_predictor',
    version='0.0.01',
    description="Demand Predictor Model (api_pred)",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False
)
