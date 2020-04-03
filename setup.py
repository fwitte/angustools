from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='angustools',
      version='0.1',
      description='ANGUS II Tools Package',
      url='http://github.com/fwitte/angustools',
      author='Francesco Witte',
      author_email='francesco.witte@web.de',
      long_description=read('README.rst'),
      license='MIT',
      packages=['angustools', 'angustools.compressedair', 'angustools.heat'],
      python_requires='>=3',
      install_requires=['numpy>=1.13.3,<2',
                        'pandas>=0.19.2,!=1.0.0,<2'])
