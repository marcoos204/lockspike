# -*- coding: utf-8 -*-
"""
Co-simulation: Spike + Punxa lock-step verification
Currently loading a OpenSBI payload image with a small
linux kernel and initramfs image
Usage:
    python -i tb_Buildroot_m.py
    >>> from cosim_pyspike import prepare_cosim, cosim_run
    >>> prepare_cosim()
    >>> cosim_run(100000)
"""
import sys
import os
from elftools.elf.elffile import ELFFile
from riscv.sim import sim_t
from riscv.cfg import cfg_t, mem_cfg_t
from riscv.debug_module import debug_module_config_t
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FW_PAYLOAD_BIN = os.path.normpath(os.path.join(SCRIPT_DIR, './payloads/fw_payload.bin'))
DTB_FILE       = os.path.normpath(os.path.join(SCRIPT_DIR, './payloads/pyspike_initramfs_noplic.dtb'))

MEM_BASE       = 0x8000_0000
MEM_SIZE       = 0x1000_0000
ENTRY_PC       = 0x8000_0000
DTB_ADDR       = 0x8200_0000
TRAMPOLINE     = 0x8070_0000
CHECKPOINT_STP = 10000000
# KERNEL_ADDR    = 0x8020_0000
# INITRAMFS_ADDR = 0x8400_0000

ISA = "rv64imafdc_zicsr_zifencei_zicntr"

spike = None
hart0 = None
step_count = 0
ilast = 0
punxa_cpu = None
punxa_hw = None
ring_buff = None

#legacy function handler
#previously we wrote only with the exposed store_uint8
#should refactor later
def write_u32(hart, addr, val):
    hart.mmu.store_uint32(addr, val) 

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

def write_trampoline(hart, cpu):
    """Sets a0=0, a1=0x82000000, jumps to 0x80000000.
       We need to do this bc Spike protects its own
       bootROM as a device instead of mapped memory"""
    insns = [
        0x00000513,  # addi  a0, x0, 0
        0x04100593,  # addi  a1, x0, 0x41
        0x01959593,  # slli  a1, a1, 25         -> a1 = 0x82000000
        0x00100293,  # addi  t0, x0, 1
        0x01f29293,  # slli  t0, t0, 31         -> t0 = 0x80000000
        0x00028067,  # jalr  x0, 0(t0)
    ]
    for i, insn in enumerate(insns):
        write_u32(hart, TRAMPOLINE + i*4, insn)
        cpu.behavioural_memory.write_i32((TRAMPOLINE + i*4) - MEM_BASE, insn)

    print(f'[*] bootROM wrote at {TRAMPOLINE:#x}: a0=0, a1={DTB_ADDR:#x} -> {ENTRY_PC:#x}')

def read_insn_at_pc(hart, cpu, pc):
    """Read the binary value of a 32-bit instruction word at PC.

    First tries Spike's MMU (which handles virtual addresses). Falls
    back to Punxa's behavioural memory for physical addresses when
    Spike's MMU rejects the access. Returns None if neither works.
    """
    # Try Spike's MMU first, but it could fail depending on the MMU state.
    # This works but it would probably be better to just create or call a VA_translate
    # function from spike to avoid doing the load.
    try:
        low_parcel = hart.mmu.load_uint16(pc)
        if (low_parcel & 0x3) != 0x3: # check for 16-bit compressed instructon
            return low_parcel
        if pc % 4 == 0: # check if the 32-bit instruction is 4-byte aligned
            return hart.mmu.load_uint32(pc) & 0xFFFFFFFF
        high_parcel = hart.mmu.load_uint16(pc + 2)
        return (high_parcel << 16) | low_parcel
    except Exception:
        pass

    # Fallback: Punxa's physical memory only for openSBI instructions
    if MEM_BASE <= pc < MEM_BASE + MEM_SIZE:
        try:
            return cpu.behavioural_memory.read_i32(pc - MEM_BASE)
        except Exception:
            pass

    return None

#fetch state data from hart0
#TODO: add fpr reg if enforced = true
def get_reg_state(hart):
    state = {}
    state['pc'] = hart.state.pc
    for i in range(32):
        state[f'x{i}'] = hart.state.XPR[i]
    state['mstatus']  = hart.get_csr(0x300)
    state['misa']     = hart.get_csr(0x301)
    state['mie']      = hart.get_csr(0x304)
    state['mtvec']    = hart.get_csr(0x305)
    state['mscratch'] = hart.get_csr(0x340)
    state['mepc']     = hart.get_csr(0x341)
    state['mcause']   = hart.get_csr(0x342)
    state['mtval']    = hart.get_csr(0x343)
    state['mip']      = hart.get_csr(0x344)
    state['mideleg']  = hart.get_csr(0x303)
    state['medeleg']  = hart.get_csr(0x302)
    state['sepc']     = hart.get_csr(0x141)
    state['scause']   = hart.get_csr(0x142)
    state['sstatus']  = hart.get_csr(0x100)
    state['satp']     = hart.get_csr(0x180)
    return state
#TODO add fpr reg if enforce = true
def print_reg_state(state, sim=""):
    print(f"--- Register state {sim} ---")
    print(f"  PC: {state['pc']:#018x}")
    for i in range(32):
        val = state[f'x{i}']
        if val != 0:
            print(f"  x{i:02d}: {val:#018x}")
    for csr in ['mstatus', 'mie', 'mtvec', 'mscratch', 'mepc', 'mcause',
                'mideleg', 'medeleg', 'sepc', 'scause', 'sstatus', 'satp']:
        val = state.get(csr, 0)
        if val != 0:
            print(f"  {csr}: {val:#018x}")

def compare_regs(spike_state, punxa_state, ignore_csrs=False):
    mismatches = []
    if spike_state['pc'] != punxa_state['pc']:
        mismatches.append(('pc', spike_state['pc'], punxa_state['pc']))
    for i in range(32):
        reg = f'x{i}'
        sv = spike_state[reg]
        pv = punxa_state[reg]
        if sv != pv:
            mismatches.append((reg, sv, pv))
    if not ignore_csrs:
        for csr in ['mstatus', 'mie', 'mtvec', 'mscratch', 'mepc', 'mcause',
                     'mideleg', 'medeleg']:
            sv = spike_state.get(csr, 0)
            pv = punxa_state.get(csr, 0)
            if sv != pv:
                mismatches.append((csr, sv, pv))
    return mismatches

def compare_mem(hart, write_list):
    mismatches = []
    #ugly satp hack to get phys load
    saved_satp = hart.get_csr(0x180)
    hart.put_csr(0x180, 0)
    saved_prv = hart.state.prv
    hart.set_privilege(3, False)  # M mode, sin translation
    for pa, va, size, punxa_val in write_list:
        try:
            if size == 1:
                spike_val = hart.mmu.load_uint8(pa)
            elif size == 2:
                spike_val = hart.mmu.load_uint16(pa)
            elif size == 4:
                spike_val = hart.mmu.load_uint32(pa)
            elif size == 8:
                spike_val = hart.mmu.load_uint64(pa)
            else:
                continue
            if punxa_val != spike_val:
                mismatches.append((f'mem_pa_{pa:#x}_from_mem_va_{va:#x}_{size}B', spike_val, punxa_val))
        except Exception:
            mismatches.append((f'mem_access_fault_{pa:#}_from_va{va:#x}', 0, 0))
    hart.put_csr(0x180, saved_satp)
    hart.set_privilege(saved_prv, False)

    return mismatches

#TODO: add fpr reg if enforced = true
def get_punxa_state(cpu):
    state = {}
    state['pc'] = cpu.pc
    for i in range(32):
        state[f'x{i}'] = cpu.reg[i] & 0xFFFFFFFFFFFFFFFF
    state['mstatus']  = cpu.csr[0x300]
    state['mie']      = cpu.csr[0x304]
    state['mtvec']    = cpu.csr[0x305]
    state['mscratch'] = cpu.csr[0x340]
    state['mepc']     = cpu.csr[0x341]
    state['mcause']   = cpu.csr[0x342]
    state['mideleg']  = cpu.csr[0x303]
    state['medeleg']  = cpu.csr[0x302]
    state['sepc']     = cpu.csr[0x141]
    state['scause']   = cpu.csr[0x142]
    state['sstatus']  = cpu.csr[0x100]
    state['satp']     = cpu.csr[0x180]
    return state


def prepare():
    global spike, hart0, step_count
    print(f'[*] Lock-step simulation config')
    print(f'[*] Firmware:   {FW_PAYLOAD_BIN}')
    print(f'[*] DTB:        {DTB_FILE}')
    print(f'[*] ISA:        {ISA}')
    print(f'[*] Memory:     {MEM_SIZE // (1024*1024)} MB @ {MEM_BASE:#x}')

    cfg = cfg_t( #spike configuration
        isa=ISA,
        priv="msu",
        mem_layout=[mem_cfg_t(MEM_BASE, MEM_SIZE)]
    )

    spike = sim_t( #initialize instance of sim_t class
        cfg=cfg,
        halted=False,
        plugin_device_factories=[],
        args=["spike"],
        dm_config=debug_module_config_t(),
    )
    hart0 = spike.get_core(0)

    #Load both binaries into mem

    fw_size = load_binary_spike(hart0, FW_PAYLOAD_BIN, MEM_BASE)

    load_binary_spike(hart0, DTB_FILE, DTB_ADDR)

    write_trampoline(hart0, punxa_cpu)

    hart0.state.pc = TRAMPOLINE
    for i in range(6):
        spike.step(1)

    assert hart0.state.pc == ENTRY_PC
    assert hart0.state.XPR[10] == 0
    assert hart0.state.XPR[11] == DTB_ADDR

    step_count = 0
    print(f'[*] Spike ready. PC={hart0.state.pc:#x}, a0={hart0.state.XPR[10]}, a1={hart0.state.XPR[11]:#x}')

def get_state():
    print_reg_state(get_reg_state(hart0), label=f"step={step_count}")

def run_steps(n, print_every=100000):
    global step_count
    for i in range(n):
        spike.step(1)
        step_count += 1
        if print_every and step_count % print_every == 0:
            print(f'  step {step_count}, PC: {hart0.state.pc:#018x}')
    print(f'[*] Done. Total steps: {step_count}, PC: {hart0.state.pc:#018x}')

def run_until(target_pc, max_steps=100_000_000):
    global step_count
    for i in range(max_steps):
        if hart0.state.pc == target_pc:
            print(f'[*] Reached {target_pc:#x} at step {step_count}')
            return True
        spike.step(1)
        step_count += 1
        if step_count % 1000000 == 0:
            print(f'  step {step_count}, PC: {hart0.state.pc:#018x}')
    print(f'[!] Did not reach {target_pc:#x} after {max_steps} steps. '
          f'PC: {hart0.state.pc:#018x}')
    return False

def checkpoint_cosim(filename_prefix='cosim'):
    """
    Save the state of the lockstep sim to checkpoint files.

    Use after prepare_cosim() and some cosim_run() steps. Typical usage:
        prepare_cosim()
        cosim_run(NUM_STEPS_TO_KERNEL_ENTRY)
        checkpoint_cosim('kernel_entry')

    Restore later in a fresh REPL session with:
        prepare_cosim()
        restore_cosim('kernel_entry')
        cosim_run(...)
    """
    
    from .checkpoint.checkpoint_spike import checkpoint_spike
    from punxa.interactive_commands import checkpoint as punxa_checkpoint
    
    ckpt_file = f'{filename_prefix}.punxa.dat'
    meta_file  = f'{filename_prefix}.cosim.meta'

    print(f'[*] Saving checkpoint to {ckpt_file}...')
    punxa_checkpoint(ckpt_file)

    #print(f'[*] Saving Spike checkpoint to {spike_file}...')
    #mem_regions = [
    #    (MEM_BASE, MEM_SIZE),    # main memory 256MB
    #    (DTB_ADDR, 0x10000),     # DTB
    #]
    #checkpoint_spike(hart0, mem_regions, spike_file, sim=spike)

    with open(meta_file, 'w') as f:
        f.write(f'step_count={step_count}\n')
    print(f'[*] Saved cosim metadata to {meta_file}')

    print(f'[*] Cosim checkpoint complete. Step count: {step_count}')


def restore_cosim(filename_prefix='cosim'):
    """
    Restore the state of both Punxa and Spike from checkpoint files.

    Requires prepare_cosim() to have been called first to initialize
    both simulators with the base configuration. This function then
    overwrites their state with the checkpointed values.

    Usage:
        prepare_cosim()
        restore_cosim('kernel_entry')
        cosim_run(...)
    """
    global step_count
    
    from .checkpoint.checkpoint_spike import restore_spike
    from punxa.interactive_commands import restore as punxa_restore

    ckpt_file = f'{filename_prefix}.punxa.dat'
    meta_file  = f'{filename_prefix}.cosim.meta'

    print(f'[*] Restoring Punxa state from {ckpt_file}...')
    punxa_restore(ckpt_file)

    print(f'[*] Restoring Spike state from {ckpt_file}...')
    restore_spike(hart0, ckpt_file, sim=spike)

    try:
        with open(meta_file, 'r') as f:
            for line in f:
                key, val = line.strip().split('=')
                if key == 'step_count':
                    step_count = int(val)
                    break
        print(f'[*] Restored step_count: {step_count}')
    except FileNotFoundError:
        print(f'[!] Metadata file {meta_file} not found, step_count not restored')

    print(f'[*] Cosim state restored at step {step_count}')


class PrintRingBuffer:
    def __init__(self, depth):
        self.buffer = deque(maxlen=depth)
    def __call__(self, *args, **kargs):
        self.buffer.append((args,kargs))
    def flush(self):
        for args, kargs in self.buffer:
            print(*args, **kargs)
        self.buffer.clear()
def cosim_step(punxa_cpu, punxa_hw, n=1, ignore_csrs=False, verbose=False, enforce=False):
    """Step both Spike and Punxa, compare after each instruction.
    """
    global step_count
    global ilast
    global ring_buff
 
    sim = punxa_hw.getSimulator()
    trap_syncs = 0
    timer_syncs = 0
    stip_syncs = 0
    intr_syncs = 0 

    TIMER_CSRS = {
        0xC00, 0xC01, 0xC02,
        0xC80, 0xC81, 0xC82,
        0xB00, 0xB02,
    }

    MTIP_BIT = 1 << 7 

    for i in range(n):
        #infinite timer in cosim fix - we need to update clint's mtime in punxa, not just the read effect
        punxa_hw.children['clint'].mtime = hart0.get_csr(0xC01)
        spike.clint.set_mtimecmp(0, 0xffffffffffffffff)

        spike_pc_before = hart0.state.pc
        punxa_pc_before = punxa_cpu.pc

        # Step Punxa: clk until INSTRET advances by 1
        instret_before = punxa_cpu.getCSR(0xC02)
        max_clks = 1000
        clks = 0
        while clks < max_clks:
            sim.clk(1)
            clks += 1
            if punxa_cpu.getCSR(0xC02) != instret_before:
                break
        
        # Step Spike
        spike.step(1)
        step_count += 1
        
        # We sync STIP (the timer interrupts) if punxa took an interrupt but Spike didn't
        punxa_mip = punxa_cpu.csr[0x344]
        spike_mip = hart0.get_csr(0x344)
        punxa_mtip = punxa_mip & MTIP_BIT
        spike_mtip = spike_mip & MTIP_BIT
        if punxa_mtip and not spike_mtip:
            now = hart0.get_csr(0xC01)
            spike.clint.set_mtimecmp(0, now) #forcing pending interrupts
            stip_syncs += 1
        elif spike_mtip and not punxa_mtip:
            now = hart0.get_csr(0xC01)
            spike.clint.set_mtimecmp(0, now)
            stip_syncs += 1

        # mret/sret detection uses read_insn_at_pc which works
        # regardless of paging, with Punxa fallback for physical PCs
        # when Spike's MMU rejects the access.
        spike_pc_after = hart0.state.pc
        spike_mtvec = hart0.get_csr(0x305)
        spike_stvec = hart0.get_csr(0x105)

        ins_before = read_insn_at_pc(hart0, punxa_cpu, spike_pc_before)
        is_mret = (ins_before == 0x30200073) if ins_before is not None else False
        is_sret = (ins_before == 0x10200073) if ins_before is not None else False
        
        spike_trapped = (spike_pc_after != spike_pc_before and
                         spike_pc_after in (spike_mtvec, spike_stvec) and
                         spike_pc_after != spike_pc_before + 2 and
                         spike_pc_after != spike_pc_before + 4 and
                         not is_mret and not is_sret)

        if spike_trapped:
            if verbose:
                print(f'  [trap] Step {step_count}: Spike trapped @ {spike_pc_before:#x} '
                      f'-> mtvec {spike_pc_after:#x}, stepping Spike again to sync')
            spike.step(1)
            trap_syncs += 1
        
        # Punxa may take an interrupt that Spike hasn't
        # taken yet. This happens because Punxa evaluates pending
        # interrupts at the end of the instruction that enables them
        # (e.g., CSRRSI sstatus |= SIE), while Spike evaluates them at
        # the start of the next instruction. When Punxa jumps to stvec
        # in the same step that enables interrupts, Spike needs an
        # extra step to catch up, just as the previous spike_trapped line
        
        punxa_pc_after = punxa_cpu.pc
        punxa_at_trap_vec = punxa_pc_after in (
            punxa_cpu.csr[0x305],  # mtvec
            punxa_cpu.csr[0x105],  # stvec
        )
        punxa_scause = punxa_cpu.csr[0x142]
        punxa_mcause = punxa_cpu.csr[0x342]
        punxa_intr = ((punxa_scause >> 63) & 1) or ((punxa_mcause >> 63) & 1)
        spike_at_trap_vec = hart0.state.pc in (spike_mtvec, spike_stvec)

        if (not spike_trapped and
            punxa_at_trap_vec and
            punxa_intr and
            not spike_at_trap_vec):
            if verbose:
                print(f'  [intr] Step {step_count}: Punxa took interrupt @ {punxa_pc_before:#x} '
                      f'-> {punxa_pc_after:#x}, stepping Spike again to sync')
            spike.step(1)
            intr_syncs += 1
        spike_state = get_reg_state(hart0)
        punxa_state = get_punxa_state(punxa_cpu)

        if verbose:
            print(f'\n=== Step {step_count} (punxa clks: {clks}) ===')
            print(f'  Executed instruction @ {spike_pc_before:#018x}')
            if spike_trapped:
                print(f'  (trap sync applied)')
            print_reg_state(spike_state, 'Spike')
            print_reg_state(punxa_state, 'Punxa')

        # Stop at kernel entry (Linux head): both simulators must be there.
        #KERNEL_ENTRY_PC = 0x80200000
        #if spike_state['pc'] == KERNEL_ENTRY_PC and punxa_state['pc'] == KERNEL_ENTRY_PC:
        #    print(f'[*] Reached kernel entry @ {KERNEL_ENTRY_PC:#x} at step {step_count}')
        #    return True, step_count, []
        
        #chechk reg mismatches
        mismatches = compare_regs(spike_state, punxa_state, ignore_csrs=ignore_csrs)
        
        #check mem mismatches if enforced=true
        if enforce and punxa_cpu.write_list:
            mem_mismatches = compare_mem(hart0, punxa_cpu.write_list)
            if mem_mismatches:
                mismatches.extend(mem_mismatches)
            punxa_cpu.write_list.clear()

        if mismatches:
            if spike_state['pc'] == punxa_state['pc']:
                ins = read_insn_at_pc(hart0, punxa_cpu, spike_pc_before)
                if ins is not None:
                    csr_field = (ins >> 20) & 0xFFF
                    funct3 = (ins >> 12) & 0x7
                    is_csr_op = funct3 in (1, 2, 3, 5, 6, 7)

                    if is_csr_op and csr_field in TIMER_CSRS:
                        for reg, sv, pv in mismatches:
                            if reg.startswith('x'):
                                reg_num = int(reg[1:])
                                punxa_cpu.reg[reg_num] = sv #here we sync at the register level. It won't change internal device registers like mtime
                        timer_syncs += 1
                        if verbose:
                            print(f'  [timer] Step {step_count}: auto-synced CSR {csr_field:#x} '
                                  f'read at {spike_pc_before:#x}')
                        continue

            print(f'\n[!] MISMATCH at step {step_count} (punxa clks: {clks}):')
            print(f'    Instruction that diverged:')
            print(f'      Spike executed @ {spike_pc_before:#018x}')
            print(f'      Punxa executed @ {punxa_pc_before:#018x}')
            if spike_trapped:
                print(f'      (trap detected, Spike stepped twice)')
            print()
            for reg, sv, pv in mismatches:
                print(f'    {reg}: spike={sv:#018x}  punxa={pv:#018x}')
            print()
            print_reg_state(spike_state, f'Spike step={step_count}')
            print()
            print_reg_state(punxa_state, f'Punxa step={step_count}')
            print(f'\n[*] Trap syncs: {trap_syncs}, Timer syncs: {timer_syncs}, stip syncs: {stip_syncs}, intr syncs: {intr_syncs})')
            print(f'\n[*] Last 10 instructions/events executed: ')
            ring_buff.flush()
            return False, step_count, mismatches
        else:
            if ((step_count % CHECKPOINT_STP) == 0):
                checkpoint_cosim('lockstep_prev_10M')

        icur = punxa_cpu.getCSR(0xC02) # 0xC02 is CSR_INSTRET
        if (icur % 10000 == 0) and (icur != ilast):
            print('ins: {:n}'.format(icur))
            ilast = icur

    print(f'[*] {n} steps compared OK. Total: {step_count} '
          f'(trap syncs: {trap_syncs}, timer syncs: {timer_syncs}, '
          f'stip syncs: {stip_syncs}, intr syncs: {intr_syncs})')
    return True, step_count, []

def punxa_write_capture(pa, size, value, va):
    global punxa_cpu
    mask = (1 << (8 * size)) -  1 #mask higher bits depending on size for false positives
    punxa_cpu.write_list.append((pa, va, size, value & mask))

def prepare_cosim():
    global punxa_cpu, punxa_hw, ring_buff
    ring_buff = PrintRingBuffer(depth=35)

    import __main__ as main_module

    main_module.buildHw()

    main_module.uart.reg_size = 1
    print('[*] Punxa UART set to reg_size=1 (generic platform)')

    memory = main_module.memory
    cpu = main_module.cpu
    mem_base = main_module.mem_base

    main_module.reallocMem(0x80000000, MEM_SIZE)
    main_module.reallocMem(DTB_ADDR, 0x10000)

    print('[*] Loading fw_payload.bin into Punxa...')
    main_module.loadProgram(memory, FW_PAYLOAD_BIN, MEM_BASE - mem_base)

    print('[*] Loading DTB into Punxa...')
    main_module.loadProgram(memory, DTB_FILE, DTB_ADDR - mem_base)

    cpu.reg[10] = 0
    cpu.reg[11] = DTB_ADDR
    cpu.min_clks_for_trace_event = 1

    try:
        main_module.loadSymbolsFromElf(cpu, OPENSBI_ELF, 0)
        print('[*] OpenSBI symbols loaded')
    except Exception as e:
        print(f'[*] No OpenSBI symbols: {e}')

    punxa_cpu = cpu
    punxa_hw = main_module.hw

    prepare()

    punxa_cpu.reg[5] = 0x80000000
    punxa_cpu.csr_write_verbose = False
    #Hot patch to the Punxa module namespace
    #Probably should rework this into some sort of flag inside punxa
    #Or just implement the ring buff function into the dummy print for punxa

    #Quick patch to the punxa's namespace injecting the ring buff
    import sys
    cpu_module = sys.modules[punxa_cpu.__module__]
    cpu_module.dummy_print = ring_buff
    cpu_module.pr = ring_buff

    punxa_cpu.setVerbose(False)
    # match Spike's machine info CSR values for cosim compatibility.
    punxa_cpu.csr[0xF11] = hart0.get_csr(0xF11)  # mvendorid
    punxa_cpu.csr[0xF12] = hart0.get_csr(0xF12)  # marchid
    punxa_cpu.csr[0xF13] = hart0.get_csr(0xF13)  # mimpid

    print('[*] Both simulators ready for co-simulation')
    print(f'[*] Firmware: {FW_PAYLOAD_BIN}')
    print(f'[*] DTB:      {DTB_FILE}')
    print(f'[*] Address map: fw_payload@{MEM_BASE:#x} DTB@{DTB_ADDR:#x}')
    print('[*] Use: cosim_run(n)  or  cosim_run(n, verbose=True)')


def cosim_run(n=1, ignore_csrs=True, verbose=False, enforce=False):
    if (enforce):
        punxa_cpu.mem_write_hook = punxa_write_capture
    else:
        punxa_cpu.mem_write_hook = None
    return cosim_step(punxa_cpu, punxa_hw, n=n, ignore_csrs=ignore_csrs, verbose=verbose, enforce=enforce)

def step_spike(n=1):
    print(f'[*] Stepping spike for {n} steps')
    spike.step(n)
    print(f'[*] Spike done')

if __name__ == "__main__":
    print(sys.argv)
    prepare()
