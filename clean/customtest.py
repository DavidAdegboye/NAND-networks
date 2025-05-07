import npn
from itertools import product
import yaml
import time
import main

with open("set-up-custom.yaml", "r") as f:
    config = yaml.safe_load(f)

true_start = time.time()

def generate_npn_classes():
    seen = set()
    for bits in product([0, 1], repeat=16):
        canonical = tuple(npn.npn_canonical_representative(list(bits)))
        seen.add(canonical)
    return seen

npn_classes = generate_npn_classes()
print(f"Total unique NPN classes for 4-input functions: {len(npn_classes)}")

with open(config["output_file"], "w") as f:
    f.write(f"New test:\n")

for npn_class in npn_classes:
    run_start = time.time()
    success = False
    attempts = 0
    while not success:
        try:
            attempts += 1
            with open(config["output_file"], "a") as f:
                f.write(f"Attempt {attempts:}\n")
            success = main.run_test({"output": [[int(entry)] for entry in npn_class]},
                                "set-up-custom.yaml")
        except KeyboardInterrupt:
            success = True
            break
        except Exception as e:
            with open(config["output_file"], "a") as f:
                f.write(f'Error "{e}" with following truth table:\n{npn_class}\n')
    run_end = time.time()
    with open(config["output_file"], "a") as f:
        f.write(f"Total time for test: {run_end - run_start} seconds.\n")
    run_start = time.time()
    success = False
    attempts = 0
    while not success:
        try:
            attempts += 1
            with open(config["output_file"], "a") as f:
                f.write(f"Attempt {attempts:}\n")
            success = main.run_test({"output": [[1-int(entry)] for entry in npn_class]},
                                "set-up-custom.yaml")
        except KeyboardInterrupt:
            success = True
            break
        except Exception as e:
            with open(config["output_file"], "a") as f:
                f.write(f'Error "{e}" with following truth table:\n{[not entry for entry in npn_class]}\n')
    run_end = time.time()
    with open(config["output_file"], "a") as f:
        f.write(f"Total time for test: {run_end - run_start} seconds.\n")
true_end = time.time()
with open(config["output_file"], "a") as f:
    f.write(f"Total time for 444 tests: {true_end - true_start} seconds.\n")
