DEFAULT_LOG_DIR=/var/log/monasca-predictor
DEFAULT_VENV_DIR=.venv

.PHONY: all log py37

all: py37 log

log: $(DEFAULT_LOG_DIR)/

py37: $(DEFAULT_VENV_DIR)/py37/

$(DEFAULT_VENV_DIR)/py37/:
	python3.7 -m venv $@
	$@/bin/pip install -U pip
	$@/bin/pip install -e .
	$@/bin/pip install -r test-requirements.txt
	@echo "Run \`source $@/bin/activate\` to start the virtual env."

$(DEFAULT_LOG_DIR)/:
	sudo mkdir -p $@
	sudo chown $(USER):$(USER) $@
	chmod -R 755 $@
