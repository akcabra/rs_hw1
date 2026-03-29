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

# Task 1d: Register File Sweep

num_registers = int(sys.argv[1]) if len(sys.argv) > 1 else 64
version = sys.argv[2] if len(sys.argv) > 2 else "original"

print(f"[{datetime.datetime.now()}] Task 1d: Num Registers = {num_registers}, Version = {version}")

cache_hierarchy = PrivateL1CacheHierarchy(l1d_size="32KiB", l1i_size="32KiB")
memory = SingleChannelDDR3_1600("7GiB")

processor = RISCV_O3_CPU(pipeline_width=2, rob_size=128, num_int_regs=num_registers, num_float_regs=num_registers)

board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy
)

if version == "optimized":
    binary_path = "./workload/scaled_dot_product_adv.bin"
else:
    binary_path = "./workload/scaled_dot_product.bin"

binary = CustomResource(binary_path)
board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
print(f"[{datetime.datetime.now()}] Simulation started: registers={num_registers}, version={version}")
simulator.run()
