import os
from setuptools import setup


__version__ = '0.0.1'
__package__ = 'kelly'

here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    required = f.read().splitlines()

with open(os.path.join(here, "README.md")) as f:
    __doc__ = f.read()


setup(
    name=__package__,
    version=__version__,
    author='joao ceia',
    author_email='joao.p.ceia@gmail.com',
    packages=[__package__],
    url='https://github.com/jpceia/kelly',
    license='nolicense',
    description="Python package for Kelly Criterion",
    long_description=__doc__,
    long_description_content_type="text/markdown",
    install_requires=required,
    tests_require=["nose", "coverage"],
    test_suite="nose.collector",
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)
