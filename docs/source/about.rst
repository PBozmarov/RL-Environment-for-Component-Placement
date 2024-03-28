About
=====


.. _pcb_problem:

The PCB Problem
----------------------------

Printed circuit boards (PCBs) are the core foundation supporting most electronic products used
today. While the PCB industry has advanced significantly over the decades, increasing the speed
and capabilities of electronic components whilst simultaneously decreasing their size, the design cycle
for manufacturing PCBs still has ample room for improvement. Despite decades of research on this
problem, the current industry procedures still require human experts to manually modify PCBs by
using placement tools in order to produce solutions.


.. _related_work:

Related Work
-----------------------

The design of a PCB is broken into 4 steps: (1) schematic design, (2) placement, (3) routing and (4)
testing. In the placement and routing stages, the designers would iteratively place and route components until specific constraints have been satisfied. The board would then be simulated and tested
for production. Today much of the PCB design load is automated by Electronic Design Automation (EDA) software packages. But while this automation has improved several processes in the PCB
production pipeline, the actual component placement and routing task requires designer (human) intervention.


.. _contribution:

Our Contribution
----------------------------------------------

This repository provides a framework for modeling the component placement problem as a reinforcement learning problem. A suite 
of environments, each with a different level of complexity, are provided to facilitate the training of reinforcement learning agents.
The most complex of these environments is an environment consiting of a grid of size height x width and a number of components of varying
heights and widths with pins upon them. The goal of the agent is to place the components on the grid in a manner which optimizes the subsequent
routing of the pins. 

The agents are trained using the Proximal Policy Optimization (PPO) algorithm. The repository provides a number of different agents which can be
trained on the environments, each of which has a different architecture for the policy network. Besides providing the functionality
of training reinfrocement learning agents, a web-app implemented using streamlit is also provided for the task of training and visualizing the
performance of different agents.


.. _contributors:

Contributors
----------

- Devesh Joshi    devesh.joshi22@imperial.ac.uk

- Samuel Kelso    samuel.kelso22@imperial.ac.uk

- Yolanda Yang    xiaoxue.yang22@imperial.ac.uk

- Pavel Bozmarov    pavel.bozmarov22@imperial.ac.uk

- Kianoosh Ashouritaklimi    kianoosh.ashouritaklimi22@imperial.ac.uk

- Joshan Dooki    joshan.dooki22@imperial.ac.uk