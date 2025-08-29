SUBDIRS := src

all: $(SUBDIRS)
$(SUBDIRS):
	+$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: all $(SUBDIRS)

run_nsys:
	nsys profile --gpu-metrics-device=all --gpu-metrics-frequency=1000 --cuda-graph-trace=node .PHONY: all $(SUBDIRS)
