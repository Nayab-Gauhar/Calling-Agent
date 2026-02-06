#!/bin/bash
# Install setuptools first to fix Python 3.13 build issues
pip install setuptools>=69.0.0
# Then install all other dependencies
pip install -r requirements.txt
