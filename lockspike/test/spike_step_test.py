from elftools.elf.elffile import ELFFile
from riscv.sim import sim_t
from riscv.cfg import cfg_t, mem_cfg_t
from riscv.debug_module import debug_module_config_t
import time

OPENSBI_BIN    = "../payloads/fw_payload.bin"
DTB_FILE       = "../payloads/pyspike_initramfs_noplic.dtb"
PAYLOAD_ADDR   = 0x8000_0000 #  opensbi boot addr

TRAMPOLINE     = 0x8070_0000  # ROM simulation: sets a0/a1, jumps to OpenSBI
DTB_ADDR       = 0x8200_0000  # DTB location in RAM

ISA = "rv64imafdc_zicsr_zifencei_zicntr"

def get_reg_state(hart):
    state = {}
    state['pc'] = hart.state.pc
    for i in range(32):
        state[f'x{i}'] = hart.state.XPR[i]
    state['mstatus'] = hart.get_csr(0x300)
    state['mepc']    = hart.get_csr(0x341)
    state['mcause']  = hart.get_csr(0x342)
    state['sepc']    = hart.get_csr(0x141)
    state['scause']  = hart.get_csr(0x142)
    state['sstatus'] = hart.get_csr(0x100)
    state['satp']    = hart.get_csr(0x180)
    return state

def print_reg_state(state, label=""):
    print(f"--- Register state {label} ---")
    print(f"  PC: {hex(state['pc'])}")
    for i in range(32):
        print(f"  x{i:02d}: {hex(state[f'x{i}'])}")
    print(f"  mstatus: {hex(state['mstatus'])}")
    print(f"  mepc:    {hex(state['mepc'])}")
    print(f"  mcause:  {hex(state['mcause'])}")
    print(f"  sepc:    {hex(state['sepc'])}")
    print(f"  scause:  {hex(state['scause'])}")
    print(f"  sstatus: {hex(state['sstatus'])}")
    print(f"  satp:    {hex(state['satp'])}")

def write_u32(hart, addr, val):
    for i in range(4):
        hart.mmu.store_uint8(addr + i, (val >> (8*i)) & 0xFF)

def load_binary_spike(hart, path, addr):
    with open(path, "rb") as f:
        data = f.read()

    print(f'  Loading payload at: {addr:#x} ({len(data):#x} bytes)')
    for i in range (0, len(data) - 7, 8):
        chunk = data[i:i+8]
        val64 = int.from_bytes(chunk, byteorder='little')
        hart.mmu.store_uint64(addr + i, val64)
    
    if ((len(data) % 8) != 0):
        for i in range (len(data) - (len(data) % 8), len(data)):
            hart.mmu.store_uint8(addr + i, data[i])
   
    return len(data)

def write_trampoline(hart):
    insns = [
        0x00000513,  # addi  a0, x0, 0
        0x04100593,  # addi  a1, x0, 0x41
        0x01959593,  # slli  a1, a1, 25     -> a1 = 0x82000000
        0x00100293,  # addi  t0, x0, 1
        0x01f29293,  # slli  t0, t0, 31     -> t0 = 0x80000000
        0x000280e7,  # jalr  x0, 0(t0)
    ]
    for i, insn in enumerate(insns):
        write_u32(hart, TRAMPOLINE + i*4, insn)
    print(f"[*] Trampoline at {hex(TRAMPOLINE)}: a0=0, a1={hex(DTB_ADDR)} -> 0x80000000")

def main():
    cfg = cfg_t(
        isa=ISA,
        priv="msu",
        mem_layout=[mem_cfg_t(0x8000_0000, 0x1000_0000)]  # 256MB
    )

    spike = sim_t(
        cfg=cfg,
        halted=False,
        plugin_device_factories=[],
        args=["spike"],
        dm_config=debug_module_config_t(),
        dtb_file=DTB_FILE,
    )
    hart0 = spike.get_core(0)

    print(f"[*] ISA: {cfg.isa}")
    print(dir(hart0))
    load_binary_spike(hart0, OPENSBI_BIN, PAYLOAD_ADDR)

    print("[*] Loading DTB...")
    load_binary_spike(hart0, DTB_FILE, DTB_ADDR)


    write_trampoline(hart0)

    hart0.state.pc = TRAMPOLINE
    print(f"[*] Starting execution at {hex(TRAMPOLINE)}")
    step = 0
    start = time.time()
    while True:
        spike.step(1)
        step += 1
        if (hart0.state.pc == 0x0000000080200000):
            now = time.time() - start
            print("Took ", step, " steps to get here! Time elapsed: ", now, " seconds")

if __name__ == "__main__":
    main()
