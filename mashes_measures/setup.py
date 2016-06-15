from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['measures'],#, 'urdf_parser_py', 'pykdl_utils', 'hrl_geom'],
    #scripts=['scripts'],
    package_dir={'': 'src'}
)

setup(**d)
