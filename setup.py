import os
import numpy
import tempfile
import subprocess
from setuptools import setup, Extension, find_packages
from distutils import log
from distutils.command.install import install


class dummy_install(install):
    """Dummay install method that doesn't let you install."""
    def finalize_options(self, *args, **kwargs):
        raise RuntimeError("This is a dummy package that cannot be installed")


ExtensionModules = [Extension('gridder', ['gridder.cpp',], include_dirs=[numpy.get_include()], libraries=['m', 'fftw3f'], extra_compile_args=['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',]),]


setup(
    cmdclass = {'install': dummy_install}, 
    name = 'dummy_package',
    version = '0.0',
    description = 'This is a dummy package to help build the gridder extensions',
    ext_modules = ExtensionModules
)
