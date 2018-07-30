# GPLATNET

**GPLATNET** is a method for discovering the connections between the nodes in a network. That is, the given
continues-valued observations from each node over time, **GPLATNET** can discover which nodes are
connected to each other.
**GPLATNET** is composed of **GP+LATNET**. The **GP** component refers to **G**aussian **P**rocess, 
and **LATNET** refers 
to **LAT**ent **NET**work. We refer to **GPLATNET** simply as **LATNET** in the paper and below. 
 
The model is introduced in the below paper:

*Variational Network Inference: Strong and Stable with Concrete Support.* 
    <b>Amir Dezfouli, Edwin V. Bonilla, Richard Nock </b>. ICML (2018)


## Inputs and outputs
We have a network with _N_ nodes and and we have _T_ observations from each node. 
Observations are denoted by matrix **Y** (with size _N_ by _T_) and the times of observations 
are denoted by vector **t** (vector with size _T_). The aim of **LATNET** is to find (i) which 
nodes are connected
to each other and (ii) what are the strengths of the connections. 

In the examples mentioned in the experiment section, the results of running
experiments are saved into several files, as below:

* _alpha.csv_. This file represents a matrix of size _N_ by _N_.  Column _i_ and 
row _j_ corresponds to ![](http://latex.codecogs.com/svg.latex?%5Calpha_%7Bij%7D) in the paper, which
is ![](http://latex.codecogs.com/svg.latex?%5Calpha) of the posterior Concrete distribution over element _ij_ of **A**. 

* _p.csv_. This is file represents a matrix of size _N_ by _N_ and entry _ij_ can be interpreted
as the probability that node _j_ is connected to node _i_. Each element of this
matrix is ![](http://latex.codecogs.com/svg.latex?%5Calpha_%7Bij%7D%20%2F%20(1%20%2B%20%5Calpha_%7Bij%7D%20)). See the paper for more description.

* _mu.csv_. This file represents a matrix of size _N_ by _N_, which is the mean of **W**, i.e.,
the mean of the Normal distribution that determines the strength of connection from node _j_ to _i_. 
That is, column _i_ and 
row _j_ corresponds to ![](http://latex.codecogs.com/svg.latex?%5Cmu_%7Bij%7D) in the paper.

* _sigma2.csv_. This is a matrix of size _N_ by _N_, which is the variance of **W**, i.e.,
the variance of the Normal distribution that determines the strength of connection from node _j_ to _i_. 
That is, column _i_ and 
row _j_ corresponds to ![](http://latex.codecogs.com/svg.latex?%5Csigma%5E2_%7Bij%7D) in the paper.

* _hyp.csv_. This file contains optimized hyper-parameters and has four elements, as follows:
  1. The variance of observation noise, corresponding to 
  ![](http://latex.codecogs.com/svg.latex?%5Csigma%5E2_y) in the paper.
  
  2. The variance of connection noise, corresponding to 
  ![](http://latex.codecogs.com/svg.latex?%5Csigma%5E2_f) in the paper.
  
  3. The length-scale of the RBF kernel.
  
  4. The variance of the RBF kernel
  

## Examples
Folder ``experiments`` contains the data and code for running the experiments
reported in the paper. There are four experiments mentioned below. Note that for the 
ease of running the experiments the data are included in the repository.

* _fun_conn.py_. In this experiment the aim is to recover which brain regions
are connected to each other, given the activity of each region over time. For 
running this experiment you can try,

    > python -m experiments.fun_conn <n_threads>

    ``n_threads`` refers to the number of threads/processes used 
    to run in parallel. The data use for the experiment 
    was downloaded from [here](http://www.fmrib.ox.ac.uk/datasets/netsim/), which is reported in
    this paper:
    Smith SM, Miller KL, Salimi-Khorshidi G, et al. 
    Network modelling methods for FMRI. Neuroimage. 2011;54(2):875-891.

* _prices.py_. Given the median prices of different suburbs in Sydney, the aim
is to recover which suburbs are connected to each other in terms of their median prices. The data was 
downloaded from [http://www.housing.nsw.gov.au](http://www.housing.nsw.gov.au).

* _gene_reg.py_. Given the activity of each gene, the aim of this experiment is to find which
genes influence each other. The data contains activity of 800 genes. Please refer to the paper 
for the source of the data.

* _gene_reg_full.py_. Given the activity of each gene, the aim of this experiment is to find which
genes influence each other. The data contains the activity of 6178 genes. Please refer to the paper 
for the source of the data.

## Baseline models and graphs
Inside the ``experiments`` folder above, there is folder called ``R``
which contains the baseline methods and graphs.  It should be clear from
file names to which experiment they belong. Note that files ``pwling.R``
 and ``mentappr.R`` are re-implemented  in R based on their corresponding Matlab codes 
 and they are used for running _PW-LiNGAM_ algorithm. 