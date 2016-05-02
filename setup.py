#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

with open('README.rst') as readme_file:
	readme = readme_file.read()

with open('HISTORY.rst') as history_file:
	history = history_file.read()

requirements = [
	"numpy >= 1.10.4",
	"pandas >= 0.17.1",
	"simplejson >= 3.8.2",
	"scipy >= 0.17.0"
]

test_requirements = [
	'pytest',
	'h5py',
	'tables'
]

from setuptools import setup, find_packages

setup(
	name='expan',
	version='0.1.5',
	description="Experiment Analysis Library",
	long_description=readme + '\n\n' + history,
	author="Zalando SE",
	author_email='octopus@zalando.de',
	url='https://github.com/zalando/expan',
	packages=find_packages(),
	package_dir={'expan':
					 'expan'},
	include_package_data=True,
	install_requires=requirements,
	license="MIT",
	zip_safe=False,
	keywords='expan',
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Developers',
		'Intended Audience :: Data Scientists',
		'License :: OSI Approved :: MIT License',
		'Natural Language :: English',
		# "Programming Language :: Python :: 2",
		# 'Programming Language :: Python :: 2.6',
		'Programming Language :: Python :: 2.7',
		# 'Programming Language :: Python :: 3',
		# 'Programming Language :: Python :: 3.3',
		# 'Programming Language :: Python :: 3.4',
		# 'Programming Language :: Python :: 3.5',
	],
	test_suite='tests',
	tests_require=test_requirements
)
