include ../../common/make.config


default:
	(cd METISLib ; make ) 
	(cd ParMETISLib ; make ) 
	(cd Programs ; make ) 

clean:
	(cd METISLib ; make realclean )
	(cd ParMETISLib ; make realclean )
	(cd Programs ; make realclean )



run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node 
