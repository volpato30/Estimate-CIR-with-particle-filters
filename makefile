.PHONY: clean
clean:
	rm -f smc_algo*.so
	rm -f smc_algo.c
	rm -rf build

.PHONY: build
build: clean
	python cython_setup.py build_ext --inplace
