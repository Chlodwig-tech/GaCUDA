.PHONY:


rastrigin:
	nvcc rastrigin.cu -o rastrigin -lnvidia-ml

tsp:
	nvcc tsp.cu -o tsp -lnvidia-ml

root_finder:
	nvcc root_finder.cu -o root_finder -lnvidia-ml

.DEFAULT_GOAL := nothing

nothing:
	@echo "No target specified. Use 'make root_finder/rastrigin/tsp' to compile."
