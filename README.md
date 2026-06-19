Lock-step execution framework for Spike, the RISC-V ISS

Lockspike is meant to be a simple but flexible Python framework for running Spike in lock-step with another simulator.
It allows you to instrument Spike with a modified PySpike framework: Instantiate a simulation object and load any binary into Spike, then control the flow of the simulation at will. Modify register values, check its values, modify CLINT state, create checkpoints of the state, inject interruptions from other sources, modify memory regions, etc.

The lockspike.py file is an implementation of a framework for the 'Punxa' simulator. It's been used to verify it's implementation up to the usershell prompt.

Users may modify lockspike.py and adapt the framework to work in Lock-step with their own DUT. Lockspike also may be used for other purposes, such as retrieving logs from the simulation state (mem regions, GPR, CSR...) in a desired format.
