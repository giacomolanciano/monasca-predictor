DEFAULT_LOG_DIR=/var/log/monasca-predictor
DEFAULT_VENV_DIR=.venv

.PHONY: all build log py37 upload-all upload-latest

all: py37

py37: $(DEFAULT_VENV_DIR)/py37/

$(DEFAULT_VENV_DIR)/py37/:
	python3.7 -m venv $@
	$@/bin/pip install -U pip
	$@/bin/pip install -e .
	$@/bin/pip install -r test-requirements.txt
	@echo "Run \`source $@/bin/activate\` to start the virtual env."

py38: $(DEFAULT_VENV_DIR)/py38/

$(DEFAULT_VENV_DIR)/py38/:
	python3.8 -m venv $@
	$@/bin/pip install -U pip
	$@/bin/pip install -e .
	$@/bin/pip install -r test-requirements.txt
	@echo "Run \`source $@/bin/activate\` to start the virtual env."

build:
	python3 -m build

upload-all:
	python3 -m twine upload --repository pypi dist/*

upload-latest:
	python3 -m twine upload --repository pypi \
		$$(find dist/ -regex ".*\.tar\.gz" -printf "%T@ %p\n" | sort -rn | head -1 | cut -d" " -f2) \
		$$(find dist/ -regex ".*\.whl" -printf "%T@ %p\n" | sort -rn | head -1 | cut -d" " -f2) \

log: $(DEFAULT_LOG_DIR)/

$(DEFAULT_LOG_DIR):
	sudo mkdir -p $@
	sudo chown $(USER):$(USER) $@
	chmod 755 $@
