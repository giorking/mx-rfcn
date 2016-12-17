try:
	import mxnet as mx
except ImportError:
	import os, sys
	curr_path = os.path.abspath(os.path.dirname(__file__))
	# sys.path.append(os.path.join("/home/tairuic/Downloads/mxnet_pkg/mxnet/python"))
	sys.path.append(os.path.join("/home/tairuic/Desktop/mxnet/python"))
	import mxnet as mx
