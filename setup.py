from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    print(requirements)

setup(
    name='profile-image-verify',
    version='1.0',
    packages=[''],
    url='https://github.com/chrissandoval916/profile-image-verify',
    license='',
    author='Chris Sandoval',
    author_email='chrissandoval916@gmail.com',
    description='Classify Profile Images For Verification Purposes',
    install_requires=requirements
)