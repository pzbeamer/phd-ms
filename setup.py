from setuptools import setup, find_packages

setup(
    name='phd-ms',
    version='1.0',
    packages=find_packages(exclude=['tests*']),
    description='Multiscale domain identification for spatial transcriptomic data.',
    author='Perry Beamer',
    author_email='perry.beamer@gmail.com'
)