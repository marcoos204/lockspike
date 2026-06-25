# -*- coding: utf-8 -*-
"""
Checkpoint and restore for a pyspike Spike instance.
This format is meant to mimic the DUT's memory serialization map
- Spike's internal MMU caches are not serialized.
- The LR/SC reservation is not serialized
"""

import os
import shutil
import zlib

from punxa.serialize import Serializer, Deserializer

# CLINT register addresses
CLINT_MTIMECMP_ADDR = 0x02004000
CLINT_MTIME_ADDR    = 0x0200BFF8
CSR_MSTATUS = 0x300
# CSRs to skip during save/restore. Writing these via put_csr crashes
# pyspike under certain conditions:
# - FPU CSRs (fflags, frm, fcsr) crash when FPU is disabled.
#   Will need to rework activation detection in order to skip them or not
# - Wide counters trigger a spike assertion when writing on them 2 times in 1 cycle.
#   The lock-step sim already syncs timer csr.

_SKIP_CSRS = {
    0x001, 0x002, 0x003,  # fflags, frm, fcsr
    0xB00, 0xB02,         # mcycle, minstret
    0xB80, 0xB82,         # mcycleh, minstreth
    0xC00, 0xC01, 0xC02,  # cycle, time, instret
    0xC80, 0xC81, 0xC82,  # cycleh, timeh, instreth
}
for i in range(3, 32):
    _SKIP_CSRS.add(0xB00 + i)   # mhpmcounter3-31
    _SKIP_CSRS.add(0xB80 + i)   # mhpmcounter3h-31h
    _SKIP_CSRS.add(0xC00 + i)   # hpmcounter3-31
    _SKIP_CSRS.add(0xC80 + i)   # hpmcounter3h-31h


def is_fpu_enabled(hart):
    try:
        mstatus = hart.get_csr(CSR_MSTATUS)
        fs = (mstatus >> 13) & 0x3
        return fs != 0
    except Exception:
        return False


# Read mem region has to convert to bytes data retrieved from 
# spike since it's an integer data type being returned
# that way we store the data as  bytes and not integers
# Maybe should add an alignment guard so that users cannot pass 
# unaligned values

def _read_mem_region(hart, base, size):
    """
    Read [base, base+size) from Spike via the MMU as bytes. 
    Size must be 4kb aligned.
    """
    out = bytearray(size)
    for offset in range(0, size, 8):
        try:
            val = hart.mmu.load_uint64(base + offset)
            out[offset:offset + 8] = val.to_bytes(8, byteorder='little') #little endian setting
        except Exception:
            pass
    return out

# Since Spike doesn't give us a method to store bytes as ints 
# We must convert back the bytes to int data 

def _write_mem_region(hart, base, data):
    """
    Write data starting at base via the MMU.
    Size must be 4kb aligned. 
    """
    size = len(data)
    for offset in range(0, size, 8):
        chunk = data[offset:offset + 8]
        val = int.from_bytes(chunk, byteorder='little')
        try:
            hart.mmu.store_uint64(base + offset, val)
        except Exception:
            pass


def checkpoint_spike(hart, mem_regions, filename='spike.ckpt', sim=None):
    """
    Save Spike state and given memory regions to a file.
    hart : the pyspike hart object
    mem_regions : list of (base, size) tuples for memory to serialize
    filename : output file path
    """
    if os.path.exists(filename):
        shutil.copyfile(filename, filename + '.bak')

    ser = Serializer(filename)

    # PC
    ser.write_i64(hart.state.pc)

    # Current privilege mode (0=U, 1=S, 3=M).
    # Without this Spike defaults to M-mode after restore which makes
    # PC diverge with punxa
    actual_prv = hart.state.prv
    #ser.write_i64(hart.state.prv)

    hart.set_privilege(3, False) # we need to set up M mode to bypass pmp protection

    # GPRs
    for i in range(32):
        ser.write_i64(hart.state.XPR[i])

    # FPRs
    fpu = is_fpu_enabled(hart)
    for i in range(32):
        if fpu:
            try:
                ser.write_i64(hart.state.FPR[i])
            except Exception:
                ser.write_i64(0)
        else:
            ser.write_i64(0)

    # All 4096 CSRs: skipped CSRs are written as zero
    for i in range(4096):
        if i == 0xFFF:
            ser.write_i64(actual_prv)
            continue
        if not fpu and i in {0x001, 0x002, 0x003}: #fpu regs
            ser.write_i64(0)
            continue
        if i in _SKIP_CSRS:
            ser.write_i64(0)
            continue
        try:
            val = hart.get_csr(i) & ((1 << 64) - 1)
        except Exception:
            val = 0
        ser.write_i64(val)
    
    #fake stack trace to align with punxa
    ser.write_int_tuple_list([], 5)

    # Memory regions
    ser.write_i64(len(mem_regions))
    for base, size in mem_regions:
        data = _read_mem_region(hart, base, size)
        zdata = zlib.compress(bytes(data))
        ser.write_i64(base - 0x80000000)
        ser.write_i64(size)
        ser.write_i64(len(zdata))
        ser.write_bytearray(zdata)

    #fake UART console to align with punxa
    ser.write_string_list([''])

    # CLINT (mtime, mtimecmp)
    if sim is not None and sim.clint is not None:
        mtime    = sim.clint.get_mtime()
        mtimecmp = sim.clint.get_mtimecmp(0)
    else:
        try:
            mtime = hart.mmu.load_uint64(CLINT_MTIME_ADDR)
        except Exception:
            mtime = 0
        try:
            mtimecmp = hart.mmu.load_uint64(CLINT_MTIMECMP_ADDR)
        except Exception:
            mtimecmp = 0

    ser.write_i64(mtime)
    ser.write_i64(mtimecmp)

    #fake Tracer pending to align with punxa
    ser.write_dictionary({})

    hart.set_privilege(actual_prv, False) # priv mode set back to its original mode

    ser.close()
    print(f'[*] Spike checkpoint saved to {filename}')


def restore_spike(hart, filename='spike.ckpt', sim=None):    
    """
    Restore a Spike hart's state from a checkpoint file.
    """
    ser = Deserializer(filename)
    
    # PC
    hart.state.pc = ser.read_i64()
 
    hart.set_privilege(3, False) # force M mode for state restoration
    actual_prv = 0

    # GPRs
    for i in range(32):
        v = ser.read_i64()
        hart.state.XPR.write(i, v)

    # FPR read, but we check later if we can write them or not
    fpu_values = [ser.read_i64() for _ in range(32)]
    
    # we now must deser. all CSR, restore mstatus and then check if we restore
    # fpu regs or not
    # CSRs

    fpu_csrs = {}

    # CSRs
    for i in range(4096):
        v = ser.read_i64()
        if i == 0xFFF:
            actual_prv = v
            continue
        if i in (0x001, 0x002, 0x003):
            fpu_csrs[i] = v
            continue
        if i in _SKIP_CSRS:
            continue
        try:
            hart.put_csr(i, v)
        except Exception:
            pass

    if is_fpu_enabled(hart):
        for i in range(32):
            try:
                hart.state.FPR.write(i, fpu_values[i])
            except Exception:
                pass
        for i, v in fpu_csrs.items():
            try:
                hart.put_csr(i, v)
            except Exception:
                pass
    # fake stack for punxa's format equivalence
    _ = ser.read_int_tuple_list(5)
    # Memory regions
    num_regions = ser.read_i64()
    for _ in range(num_regions):
        base = ser.read_i64()
        size = ser.read_i64()
        csize = ser.read_i64()
        zdata = ser.read_bytearray(csize)
        data = zlib.decompress(zdata)
        _write_mem_region(hart, base + 0x80000000, data)

    # fake uart for punxa's format equivalence
    _ = ser.read_string_list()
    # CLINT
    mtime    = ser.read_i64()
    mtimecmp = ser.read_i64()
    if sim is not None and sim.clint is not None:
        sim.clint.set_mtime(mtime)
        sim.clint.set_mtimecmp(0, mtimecmp)
    else:
        try:
            hart.mmu.store_uint64(CLINT_MTIME_ADDR, mtime)
        except Exception:
            pass
        try:
            hart.mmu.store_uint64(CLINT_MTIMECMP_ADDR, mtimecmp)
        except Exception:
            pass
    #fake tracer for punxa's format equivalence
    _ = ser.read_dictionary()
    #restore correct prv mode
    hart.set_privilege(actual_prv, False)

    ser.close()
    print(f'[*] Spike checkpoint restored from {filename}')
