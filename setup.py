from setuptools import setup, find_namespace_packages

setup(
    name="caseva",
    version="0.0.1",
    description="Casadi/Coles Extreme Value Analysis.",
    packages=find_namespace_packages(include=['caseva*']),
    install_requires=[]
)