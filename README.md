# Lockspike: Lock-step execution Python framework for Spike, the RISC-V ISS

Lockspike is meant to be a simple but flexible Python framework for running Spike in lock-step with another simulator. Mainly, it's used to catch divergences between simulators, using Spike as the golden model.
Lockspike allows you to instrument Spike with a modified PySpike framework: Instantiate a simulation object and load any binary into Spike, then control the flow of the simulation at will. Modify register values, check its values, modify CLINT state, create checkpoints of the state, inject interrupts from other sources, modify memory regions, etc.

The lockspike.py file is an implementation of a framework for the ['Punxa'](https://github.com/marcoos204/punxa) simulator. It's been used to verify its implementation up to the user shell prompt. It's also able to provide a checkpoint functionality described at  lockspike/checkpoint/checkpoint.py, that enables Spike to store its entire simulation state (currently, in a format equivalent to Punxa's). Finally, it comes with a script that creates a single Spike simulation instance and loads a kernel payload into it, as a test for the functionality of the framework in lockstep/spike_step_test.py

Users may modify lockspike.py and adapt the framework to work in lock-step with their own DUT. Lockspike also may be used for other purposes, such as retrieving logs from the simulation state (mem regions, GPR, CSR...) in a desired format.

## How to build:

Getting the source Code:

```bash
$ git clone --recurse-submodules https://github.com/marcoos204/lockspike
$ cd lockspike
```

Then create a virtual environment

```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

Then install all dependencies for both PySpike and Punxa

```bash
(.venv) $ python -m pip install -r requirements.txt
```

## Using the framework:

The lockspike framework is a Python script specifically tailored for the Punxa simulator, but users may modify it as they like in order to adapt it for their DUT.

If you want to test the developed framework for Punxa, in order to see how it works and base off your work from that script:

```bash
(.venv) $ cd punxa/test/buildroot
(.venv) $ python -i tb_Buildroot.py
```

```python
>>> import sys
>>> sys.path.append("~/lockspike") #your path to the project
>>> from lockspike.lockspike import prepare_cosim, cosim_run
>>> prepare_cosim() #loads the payload and configures the same initial state for both simulators
>>> cosim_run(1000) #run 1000 steps in lockstep
```

Then, if you want to check how the mismatch detection works:

```python
>>> from lockspike.lockspike import punxa_cpu
>>> punxa_cpu.reg[31] = 0xBADDBEEF
>>> cosim_run(1)
```

```text
[!] MISMATCH at step 1001 (punxa clks: 6):
    Instruction that diverged:
      Spike executed @ 0x000000008000006c
      Punxa executed @ 0x000000008000006c

    x31: spike=0x0000000000000000  punxa=0x00000000baddbeef
    --- Register state Spike step=1001 ---
    PC: 0x0000000080000070
    x01: 0x0000000080000010
    x05: 0x0000000080031918
    ...
```

And, if you want to test the checkpoint functionality:

```python
>>> from lockspike.lockspike import prepare_cosim, cosim_run, checkpoint_cosim
>>> checkpoint_cosim('test')
```

```text
[*] Saving checkpoint to test.punxa.dat...
[*] Saved cosim metadata to test.cosim.meta
[*] Cosim checkpoint complete. Step count: 1000
```

```python
>>> quit()
```

Then, from a new Python REPL session:

```python
>>> import sys
>>> sys.path.append("~/lockspike")
>>> from lockspike.lockspike import prepare_cosim, cosim_run, checkpoint_cosim, restore_cosim
>>> prepare_cosim()
>>> restore_cosim('test')
```

```text
[*] Restoring Punxa state from test.punxa.dat...
[*] Restoring Spike state from test.punxa.dat...
[*] Spike checkpoint restored from test.punxa.dat
[*] Restored step_count: 1000
[*] Cosim state restored at step 1000
```

You can also modify Spike values at register, memory, or even the CLINT device registers (mtime, mtimecmp) with the functions provided by the lockspike.py script. You can check all the available functions callable from Python of Spike on pyspike/src/main/cpp/py_module.cc

If you want to test the script that handles the singular Spike instiantation via Python:

```bash
$ (.venv) cd lockspike/test/
$ (.venv) python spike_step_test.py
```

You'll be able to see a Spike simulation instance load from Python up to the kernel space, and then perform commands within the bundled initramfs.