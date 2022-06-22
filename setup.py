import setuptools

with open("requirements.txt", "r") as rf:
    requirements = [line.strip() for line in rf]

setuptools.setup(
    name="model-trainer",
    author="erudic",
    version="0.1.2",
    email="e.rudic@gmail.com",
    description="Helper training package for easier cross validation",
    packages=setuptools.find_packages(),
    install_requires=requirements
)
