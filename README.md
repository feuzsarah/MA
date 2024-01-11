# Master Thesis

# The Base Code
Our implementation is based on the code provided and developed by
<pre> 
@inproceedings{palechor2023openset,
	author       = {Palechor, Andres and Bhoumik, Annesha and G\"unther, Manuel},
	booktitle    = {Winter Conference on Applications of Computer Vision (WACV)},
	title        = {Large-Scale Open-Set Classification Protocols for {ImageNet}},
	year         = {2023},
	organization = {IEEE/CVF}
}</pre>
Please check out the base code in the following repository: https://github.com/AIML-IfI/openset-imagenet-comparison

# Remarks (Addition Negative Sample Methods)
The basic structure of this code (and how to use it) remains the same as in the base code. However, the code adaption in this repository has only been used and tested for algorithm = 'threshold' and loss = 'entropic'. Additionally, the different negative samples methods for the training procedure can be configured in the config-file neg_samples.yaml. In contrast to the base code, one also has to assign an experiment number to each experiment (in the threshold.yaml and the neg_samples.yaml) for training and evaluation.

The train command must be adapted to include the current version of the neg_samples config-file, as illustrated below: 
<pre>
python openset_imagenet/script/train.py config/threshold.yaml config/neg_samples.yaml 2 -g
</pre>
