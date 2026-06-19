from elftools.elf.elffile import ELFFile
from riscv.sim import sim_t
from riscv.cfg import cfg_t, mem_cfg_t
from riscv.debug_module import debug_module_config_t
import time

OPENSBI_ELF    = "/home/marcos/Desktop/Uni/TDR/opensbi/build/platform/generic/firmware/fw_jump.elf"
DTB_FILE       = "/home/marcos/Desktop/Uni/TDR/pyspike_initramfs.dtb"
KERNEL_BIN     = "/home/marcos/Desktop/Uni/TDR/buildroot/output/images/Image"
INITRAMFS      = "/home/marcos/Desktop/Uni/TDR/buildroot/output/images/rootfs.cpio"

TRAMPOLINE     = 0x8070_0000  # ROM simulation: sets a0/a1, jumps to OpenSBI
DTB_RAM        = 0x8200_0000  # DTB location in RAM
KERNEL_ADDR    = 0x8020_0000  # fw_jump hands off here
INITRAMFS_ADDR = 0x8400_0000  # must match linux,initrd-start in DTB

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

def compare_regs(spike_state, punxa_state):
    """Compare Spike vs Punxa register states. Returns list of (reg, spike_val, punxa_val)."""
    mismatches = []
    for reg, spike_val in spike_state.items():
        punxa_val = punxa_state.get(reg)
        if punxa_val is None:
            continue
        if spike_val != punxa_val:
            mismatches.append((reg, spike_val, punxa_val))
    return mismatches

def write_u32(hart, addr, val):
    for i in range(4):
        hart.mmu.store_uint8(addr + i, (val >> (8*i)) & 0xFF)

def load_elf(hart, elf_path):
    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        for seg in elf.iter_segments():
            if seg.header.p_type != "PT_LOAD":
                continue
            vaddr, data = seg.header.p_vaddr, seg.data()
            print(f"  PT_LOAD: {hex(vaddr)} ({hex(seg.header.p_filesz)} bytes)")
            for i, byte in enumerate(data):
                hart.mmu.store_uint8(vaddr + i, byte)
            for i in range(seg.header.p_filesz, seg.header.p_memsz):
                hart.mmu.store_uint8(vaddr + i, 0)

def load_binary(hart, path, addr):
    with open(path, "rb") as f:
        data = f.read()
    print(f"  Binary: {hex(addr)} ({hex(len(data))} bytes)")
    for i, b in enumerate(data):
        hart.mmu.store_uint8(addr + i, b)
    return len(data)

def load_dtb(hart, dtb_path):
    with open(dtb_path, "rb") as f:
        data = f.read()
    for i, b in enumerate(data):
        hart.mmu.store_uint8(DTB_RAM + i, b)
    print(f"[*] DTB loaded at {hex(DTB_RAM)} ({len(data)} bytes)")

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
    print(f"[*] Trampoline at {hex(TRAMPOLINE)}: a0=0, a1={hex(DTB_RAM)} -> 0x80000000")

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
    print(f"[*] Memory: 256MB @ 0x80000000")
    print("[*] Loading OpenSBI...")
    print(dir(hart0))
    load_elf(hart0, OPENSBI_ELF)

    print("[*] Loading kernel...")
    load_binary(hart0, KERNEL_BIN, KERNEL_ADDR)

    print("[*] Loading initramfs...")
    initramfs_size = load_binary(hart0, INITRAMFS, INITRAMFS_ADDR)
    print(f"    initrd-end = {hex(INITRAMFS_ADDR + initramfs_size)}")

    print("[*] Loading DTB...")
    load_dtb(hart0, DTB_FILE)

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
        
        #if step % 100000000== 0:
        #    print_reg_state(get_reg_state(hart0), label=f"step={step}")


        # ── Co-simulation comparison point ────────────────────────────────
        # After each step, get Spike's state and compare with Punxa.
        # Uncomment and plug in punxa.step() + punxa.get_reg_state() when ready.
        #
        # spike_state = get_reg_state(hart0)
        # punxa.step()
        # punxa_state = punxa.get_reg_state()
        # mismatches = compare_regs(spike_state, punxa_state)
        # if mismatches:
        #     print(f"[!] MISMATCH at step {step}:")
        #     for reg, sv, pv in mismatches:
        #         print(f"    {reg}: spike={hex(sv)} punxa={hex(pv)}")
        #     print_reg_state(spike_state, label=f"spike step={step}")
        #     break

if __name__ == "__main__":
    main()
