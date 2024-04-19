from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as req:
        return req.read().splitlines()

setup(
    name='MLFeatures4Materials',
    version='0.1.0',
    packages=find_packages(),
    install_requires=read_requirements(),
    author='Timo Sommer',
    description='ML Features for Materials and Molecules',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TimoSommer/MLFeatures4Materials'
)
