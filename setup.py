from setuptools import find_packages, setup

setup(
    name='on_track_sysid',
    version='2.0.0',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'scipy'],
    extras_require={'on_track': ['torch'], 'test': ['pytest']},
    python_requires='>=3.8',
    maintainer='ForzaETH',
    maintainer_email='edoardo.ghignone@pbl.ee.ethz.ch',
    description='Lateral tire identification (Pacejka) for scaled racecars: on-track and steady-state methods',
    license='MIT',
)
