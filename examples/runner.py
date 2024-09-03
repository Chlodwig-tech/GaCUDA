import subprocess
import sys
import datetime
import os

problem = sys.argv[1]

sizes      = [2 ** (10 + i) for i in range(6)]
mutations  = [1, 5, 10, 20]
crossovers = [10, 25, 50, 75]

for size in sizes:
    for mutation in mutations:
        for crossover in crossovers:
            filename = f"results/{problem}/output-{size}-{mutation}-{crossover}.txt"
            if not (os.path.exists(filename) and os.path.getsize(filename) > 0):
                command = [f'examples/{problem}', f'{size}', f'{mutation}.0', f'{crossover}.0']
                print(command, datetime.datetime.now().strftime("%H:%M:%S"))
                subprocess.run(command)
            else:
                print(f"Skipping as {filename} already exists and is not empty.")
