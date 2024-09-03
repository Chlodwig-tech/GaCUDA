.PHONY:


rastrigin:
	nvcc rastrigin.cu -o rastrigin

tsp:
	nvcc tsp.cu -o tsp

root_finder:
	nvcc root_finder.cu -o root_finder

.DEFAULT_GOAL := nothing

nothing:
	@echo "No target specified. Use 'make root_finder/rastrigin/tsp' to compile."
