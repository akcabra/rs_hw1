from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import PrivateL1CacheHierarchy
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import obtain_resource
from gem5.simulate.simulator import Simulator
from gem5.resources.resource import CustomResource
from cpuO3_model import RISCV_O3_CPU
from cpuInORD_model import RiscV_InOrder_CPU
import datetime
import sys

# Task 1c: Pipeline Width Sweep
# Usage: gem5.opt --outdir=OUT_DIR cpu_benchmark1c.py PIPELINE_WIDTH VERSION
# Example: gem5.opt --outdir=test_log_w2_orig cpu_benchmark1c.py 2 original

# Parse arguments
pipeline_width = int(sys.argv[1]) if len(sys.argv) > 1 else 2
version = sys.argv[2] if len(sys.argv) > 2 else "original"

print(f"[{datetime.datetime.now()}] Task 1c: Pipeline Width = {pipeline_width}, Version = {version}")

cache_hierarchy = PrivateL1CacheHierarchy(l1d_size="32KiB", l1i_size="32KiB")
memory = SingleChannelDDR3_1600("7GiB")

# Create O3CPU with configurable pipeline width and fixed ROB size (128)
processor = RISCV_O3_CPU(pipeline_width=pipeline_width, rob_size=128)

board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy
)

# Select workload based on version
if version == "optimized":
    binary_path = "./workload/scaled_dot_product_adv.bin"
else:
    binary_path = "./workload/scaled_dot_product.bin"

binary = CustomResource(binary_path)
board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
print(f"[{datetime.datetime.now()}] Simulation started: width={pipeline_width}, version={version}")
simulator.run()

print(f"[{datetime.datetime.now()}] Simulation completed: width={pipeline_width}, version={version}")