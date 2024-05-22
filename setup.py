from setuptools import setup, find_packages

setup(
    name='hyperbolic-optics',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'scipy',
        # Add any other dependencies here
    ],
    author='Mark Cunningham',
    author_email='m.cunningham.2@research.gla.ac.uk',
    description='4x4 Transfer Matrix Method for Anisotropic Multilayer Structures, with Mueller Matrix Calculations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MarkCunningham0410/hyperbolic-optics',
)