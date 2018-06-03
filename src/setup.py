from setuptools import setup

setup(
    name='gdayf',
    version='1.1.2',
    packages=['gdayf', 'gdayf.ui', 'gdayf.conf', 'gdayf.core', 'gdayf.logs', 'gdayf.common', 'gdayf.models',
              'gdayf.metrics', 'gdayf.handlers', 'gdayf.workflow', 'gdayf.normalizer', 'gdayf.persistence'],
    package_dir={'': 'branches/1.1.2-mrazul/src'},
    url='',
    license='',
    author='e2its',
    author_email='e2its.es@gmail.com',
    description=''
)
