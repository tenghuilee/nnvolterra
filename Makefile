
# Make Python-C library

default: all

compile: npxconv

all: result_dir npxconv_all oconv_all mnist_all

PYTHON = PYTHONPATH="$(BUILD_LIB_PATH)" python 
PYTEST = PYTHONPATH="$(BUILD_LIB_PATH)" py.test

BUILD_LIB_PATH=$(abspath $(wildcard build/lib.*))

RESULT_PATH = result

HACK_LAYER = 3
DRAW_ENG = 0.50
DRAW_TRAIN_EPHO = 4
OCONV_RAND = 'g'

npxconv: npxconv_setup.py npxconv.cpp xconvolution.hpp
	$(PYTHON) npxconv_setup.py build -f

npxconv_test: npxconv_test.py
	$(PYTEST) npxconv_test.py

npxconv_all: npxconv npxconv_test

perturbation_draw: perturbation_draw.py
	$(PYTHON) perturbation_draw.py

oconv_rank_draw: oconv_rank_draw.py
	$(PYTHON) oconv_rank_draw.py --rand $(OCONV_RAND)

oconv_rank: result_dir record_utils.py oconv_rank.py 
	$(PYTHON) oconv_rank.py --rand $(OCONV_RAND)

oconv_all: perturbation_draw

	@echo "Gaussian distribution"
	$(PYTHON) oconv_rank.py --rand 'g'
	$(PYTHON) oconv_rank_draw.py --rand 'g'

	@echo "Uniform distribution"
	$(PYTHON) oconv_rank.py --rand 'u'
	$(PYTHON) oconv_rank_draw.py --rand 'u'

mnist_train: result_dir mnist_train.py mnist_module.py
	@$(PYTHON) mnist_train.py 

mnist_hack: result_dir mnist_hack.py npxconv.py mnist_module.py
	@$(PYTHON) mnist_hack.py --hack_layer $(HACK_LAYER) --eng $(DRAW_ENG) --train_epho $(DRAW_TRAIN_EPHO)

mnist_draw: result_dir mnist_draw.py
	@$(PYTHON) mnist_draw.py  --hack_layer $(HACK_LAYER)

mnist_hack_draw: mnist_hack mnist_draw 

 
plot_all: oconv_draw 
	@echo "Gaussian distribution"
	$(PYTHON) oconv_rank_draw.py --rand 'g'
	@echo "Uniform distribution"
	$(PYTHON) oconv_rank_draw.py --rand 'u'
	@echo "vconv"
	$(PYTHON) conv_multinomial.py

hack_network: hack_network.py record_utils.py
	$(PYTHON) hack_network.py --rand $(OCONV_RAND)

hack_network_draw: result_dir hack_network_draw.py plot_utils.py record_utils.py
	$(PYTHON) hack_network_draw.py --rand $(OCONV_RAND)


mnist_all: mnist_train mnist_hack mnist_draw

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
