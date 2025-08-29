nsys profile --cuda-graph-trace=node -o BlackScholesMonolithic.nsys-rep ./BlackScholesMonolithic 1000000 256 1024 0
nsys profile --cuda-graph-trace=node -o BlackScholesKernelSplitTweak.nsys-rep ./BlackScholesKernelSplitTweak 1000000 256 1024 0
nsys profile --cuda-graph-trace=node -o BlackScholesIndependentKernelsWithGraph_V2.nsys-rep ./BlackScholesIndependentKernelsWithGraph_V2 1000000 256 1024 0
nsys profile --cuda-graph-trace=node -o BlackScholesIndependentKernelsWithGraph.nsys-rep ./BlackScholesIndependentKernelsWithGraph 1000000 256 1024 0 