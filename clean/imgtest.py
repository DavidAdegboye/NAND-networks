import yaml
import time
import main

with open("set-up.yaml", "r") as f:
    config = yaml.safe_load(f)
with open(config["output_file"], "w") as f:
    f.write(f"New test:\n")
true_start = time.time()
sigmas = {"beta_sampler": [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
          "normal_sampler1": [1, 2, 3, 4, 5, 6],
          "normal_sampler2": [0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 2.5]}
ALL_SIGMAS = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
            0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
ALL_KS = [1.0, 1.0, 1.0, 0.995, 0.99, 0.98, 0.97, 0.955, 0.94, 0.92, 0.91,
        0.9, 0.85, 0.75, 0.65, 0.5, 0.39, 0.32, 0.27, 0.23,
        0.205, 0.18, 0.17, 0.155, 0.14, 0.13, 0.12, 0.11]
ks = {s:k for (s,k) in zip(ALL_SIGMAS, ALL_KS)}
distributions = ["beta_sampler", "normal_sampler1", "normal_sampler2"]

architectures = [[2048], [2048, 1024]]
min_gatess = [[0, 0, 0], [1312, 2046, 512, 10]]
timeouts = [120, 240]
mgupcs = [0, 1]

for arch, min_gates, mgupc, timeout in zip (architectures, min_gatess, mgupcs, timeouts):
    run_start = time.time()
    main.run_test({"min_gates_used_penalty_coeff": mgupc,
            "min_gates": min_gates,
            "pool_filters": [],
            "architecture": arch,
            "timeout": timeout}, "set-up.yaml")
    run_end = time.time()
    with open(config["output_file"], "a") as f:
        f.write(f"Total time for test: {run_end - run_start} seconds.\n")
true_end = time.time()
with open(config["output_file"], "a") as f:
    f.write(f"Total time for 20 tests: {true_end - true_start} seconds.\n")
