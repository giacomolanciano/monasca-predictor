.PHONY: all py37

all: py37

py37: .venv/py37

.venv/py37:
	python3.7 -m venv $@
	$@/bin/pip install -U pip
	$@/bin/pip install -e .
	$@/bin/pip install -r test-requirements.txt
	@echo "Run \`source $@/bin/activate\` to start the virtual env."
