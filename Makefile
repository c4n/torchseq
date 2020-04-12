# Check that source code meets quality standards
quality:
	black --check --line-length 119 --target-version py36 tests src
	isort --check-only --recursive tests src
	flake8 tests src


# Format source code automatically
style:
	black --line-length 119 --target-version py36 tests src
	# isort --recursive tests src

# Check syntax
syntax:
	flake8 ./src --count --select=E9,F63,F7,F82 --show-source --statistics
	#flake8 ./src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics


# Run tests for the library
test:
	pytest

# Run tests for the library
testall:
	RUN_SLOW=1 pytest


