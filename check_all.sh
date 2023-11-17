#! /bin/sh
set -e 
echo "black check"
black . --check
echo "mypy check"
#mypy HamilToniQ
echo "pylint check"
#pylint tensorcircuit tests examples/*.py
echo "pytest check"
pytest -cov=HamilToniQ -vv -W ignore::DeprecationWarning
echo "all checks passed, congratulation! 💐"
