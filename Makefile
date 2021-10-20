
# Make Python-C library

default: compile

compile: npxconv

all: result_dir npxconv_all oconv_all mnist_all

PYTHON = PYTHONPATH="$(BUILD_LIB_PATH)" python 
PYTEST = PYTHONPATH="$(BUILD_LIB_PATH)" py.test

BUILD_LIB_PATH=$(abspath $(wildcard build/lib.*))

RESULT_PATH = result

npxconv: npxconv_setup.py npxconv.cpp xconvolution.hpp
	$(PYTHON) npxconv_setup.py build -f

npxconv_test: npxconv_test.py
	$(PYTEST) npxconv_test.py

npxconv_all: npxconv npxconv_test

oconv_draw: oconv_draw.py
	$(PYTHON) oconv_draw.py

oconv_rank_draw: oconv_rank_draw.py
	$(PYTHON) oconv_rank_draw.py

oconv_rank: oconv_rank.py result_dir
	$(PYTHON) oconv_rank.py

oconv_all: oconv_rank oconv_rank_draw oconv_draw

mnist_train: result_dir mnist_train.py mnist_module.py
	@$(PYTHON) mnist_train.py

mnist_hack: result_dir mnist_hack.py npxconv.py mnist_module.py
	@$(PYTHON) mnist_hack.py

mnist_draw: result_dir mnist_draw.py
	@$(PYTHON) mnist_draw.py 
	@-xelatex result/mnist_preview.tex

mnist_try: mnist_try.py
	@$(PYTHON) mnist_try.py
	
mnist_hack_draw: mnist_hack mnist_draw 

mnist_all: mnist_train mnist_try mnist_hack mnist_draw

pack_all:
	tar -czvf nnvolterra.tar.gz mnist_*.py npxconv_*.py oconv_*.py \
	xconvolution.hpp npxconv.* tensordec.py shape_check.py \
	README.md Makefile

result_dir:
	@if [ ! -d "$(RESULT_PATH)" ]; then \
		mkdir $(RESULT_PATH); \
	fi \

clean:
	@rm -rf build
	@rm *.so || true
