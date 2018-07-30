util.ExpLogger - DEBUG - started generating samples
util.ExpLogger - DEBUG - finished generating samples
util.ExpLogger - DEBUG - prior lambda=0.500000; posterior lambda=0.666667; variational learning rate=0.010000; hyper learning rate=0.000100
util.ExpLogger - DEBUG - iteration 0 started
util.ExpLogger - DEBUG - optimizing variational parameters
util.ExpLogger - DEBUG - elbo at local iter 0 and glocal iter 0 is -3284.307605 (KL norm=0.000000; KL logistic=4.789663, ell=-3279.517942) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - elbo at local iter 2 and glocal iter 0 is -3270.976376 (KL norm=0.071105; KL logistic=4.757540, ell=-3266.147731) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - elbo at local iter 4 and glocal iter 0 is -3264.854927 (KL norm=0.240575; KL logistic=4.667026, ell=-3259.947327) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - optimizing hyper parameters
util.ExpLogger - DEBUG - started optimizing hyper parameters
util.ExpLogger - DEBUG - elbo at local iter 0 and glocal iter 0 is -3259.125663 (KL norm=0.357013; KL logistic=4.515805, ell=-3254.252845) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - elbo at local iter 2 and glocal iter 0 is -3256.922094 (KL norm=0.357013; KL logistic=4.542051, ell=-3252.023030) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - elbo at local iter 4 and glocal iter 0 is -3260.965371 (KL norm=0.357013; KL logistic=4.881199, ell=-3255.727159) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - iteration 1 started
util.ExpLogger - DEBUG - optimizing variational parameters
util.ExpLogger - DEBUG - elbo at local iter 0 and glocal iter 1 is -3262.758054 (KL norm=0.357013; KL logistic=4.929710, ell=-3257.471331) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - elbo at local iter 2 and glocal iter 1 is -3247.233511 (KL norm=0.609130; KL logistic=4.772012, ell=-3241.852369) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - elbo at local iter 4 and glocal iter 1 is -3238.896871 (KL norm=0.879296; KL logistic=5.211586, ell=-3232.805989) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - optimizing hyper parameters
util.ExpLogger - DEBUG - started optimizing hyper parameters
util.ExpLogger - DEBUG - elbo at local iter 0 and glocal iter 1 is -3229.750336 (KL norm=1.023080; KL logistic=4.903231, ell=-3223.824025) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - elbo at local iter 2 and glocal iter 1 is -3227.989708 (KL norm=1.023080; KL logistic=4.741177, ell=-3222.225451) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - elbo at local iter 4 and glocal iter 1 is -3230.845665 (KL norm=1.023080; KL logistic=4.601401, ell=-3225.221184) with 0 nan in grads (error= 0). 
util.ExpLogger - DEBUG - alpha:[[ 1.          0.90916632  0.91361671  0.90644704  0.90575387  0.90538554
   0.90918446  0.90679022  0.91638469  0.9052752 ]
 [ 0.92641698  1.          0.91372328  0.90761755  0.90706375  0.90746977
   0.91118328  0.91063624  0.90535634  0.90886227]
 [ 0.90904307  0.90694294  1.          0.90758837  0.90743789  0.90892042
   0.90890024  0.90743644  0.90615558  0.90624677]
 [ 0.90701488  0.90803987  0.90703956  1.          0.90997686  0.90761504
   0.90749315  0.91135727  0.90836204  0.9179515 ]
 [ 0.9093979   0.90584704  0.91104416  0.90525534  1.          0.9059765
   0.90687297  0.91117111  0.9056281   0.90674383]
 [ 0.90670572  0.91224633  0.90753615  0.90711595  0.91259077  1.
   0.91106114  0.90749176  0.90601081  0.90974155]
 [ 0.90773424  0.9052477   0.90609858  0.90795838  0.90977283  0.9075965
   1.          0.9074259   0.90521328  0.91523326]
 [ 0.90838597  0.90836017  0.9153153   0.91116623  0.90557187  0.91611321
   0.91095271  1.          0.91619706  0.90651665]
 [ 0.90571824  0.90673965  0.90579938  0.90628693  0.90668617  0.90518272
   0.90811782  0.90737481  1.          0.90643368]
 [ 0.9109473   0.91666821  0.91238861  0.9076532   0.90849908  0.90461754
   0.91262066  0.91670602  0.91708595  1.        ]]
util.ExpLogger - DEBUG - mu:[[ 0.          0.09548025  0.09750703 -0.08196748 -0.07160989 -0.06273969
  -0.0563325   0.03112177 -0.07720514 -0.08969107]
 [ 0.09719278  0.          0.02803666 -0.02452334 -0.04465203 -0.0405453
  -0.06422295  0.07499739 -0.02116033 -0.07952   ]
 [ 0.09596888  0.0027854   0.         -0.09583835  0.05185914 -0.00016685
   0.08232076  0.06051983 -0.08311664 -0.05802717]
 [-0.00595022 -0.01574364 -0.01560594  0.          0.00542007 -0.0949124
  -0.02674451 -0.05303976  0.00307282 -0.06683522]
 [-0.09104702 -0.06633834  0.05178192 -0.02761293  0.         -0.03734309
   0.04660613  0.03353461 -0.08082641 -0.07921043]
 [-0.01072907 -0.04685527  0.07249128 -0.09434573  0.049904    0.
   0.03691967 -0.04194166 -0.05412502  0.03005281]
 [-0.03159508 -0.07444101  0.08343643 -0.06428793  0.01173101 -0.03286659
   0.         -0.05309411 -0.02070303 -0.09768952]
 [ 0.07217684  0.00142613  0.06886754 -0.05717612  0.04805505 -0.04551014
  -0.01464107  0.          0.04060064 -0.07125442]
 [-0.0758671  -0.05352886 -0.08184706 -0.07233626 -0.08618708 -0.0862682
  -0.0092639  -0.03056139  0.         -0.09425404]
 [-0.05630041 -0.03755658 -0.00953711 -0.04290982 -0.05599578  0.05171597
  -0.09576338 -0.06551164 -0.01283535  0.        ]]
util.ExpLogger - DEBUG - sigma2:[[ 0.2         0.18137195  0.18158483  0.18129641  0.18122975  0.18121494
   0.18213265  0.18118132  0.18225831  0.18130406]
 [ 0.18259839  0.2         0.18254975  0.18417081  0.18140459  0.18629562
   0.18171783  0.18287535  0.18108343  0.18159836]
 [ 0.18127331  0.18195401  0.2         0.18146613  0.18169087  0.18101626
   0.18170019  0.18209535  0.18119424  0.18109821]
 [ 0.18125567  0.1835881   0.18207465  0.2         0.18309099  0.18161527
   0.1813268   0.18173041  0.18138888  0.18435703]
 [ 0.18162064  0.18135992  0.18215792  0.18108953  0.2         0.18146265
   0.18130859  0.18122326  0.18120176  0.18145051]
 [ 0.1813616   0.1863269   0.18123347  0.1812513   0.18144909  0.2
   0.18231373  0.18094616  0.18124318  0.18136543]
 [ 0.181334    0.18151945  0.1812052   0.18208225  0.1817183   0.18123476
   0.2         0.18143335  0.18096205  0.18152009]
 [ 0.18160806  0.18183784  0.18225783  0.18182617  0.18097015  0.1819711
   0.18179251  0.2         0.18252079  0.18211107]
 [ 0.18116947  0.18159124  0.18102825  0.18149677  0.18143     0.18156016
   0.18131707  0.1812546   0.2         0.18141129]
 [ 0.18210539  0.18250844  0.1828717   0.18245132  0.18171912  0.1831873
   0.18156486  0.18168355  0.18283126  0.2       ]]
util.ExpLogger - DEBUG - hyp:[  1.99895870e-01   9.99091712e-05   1.00100076e-02   2.00197429e+00]
