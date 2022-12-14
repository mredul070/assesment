?	b?1???9@b?1???9@!b?1???9@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'b?1???9@n??S??1z??9y	,@Il??+?&@r0*	?|?5^RV@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??S ?g??!ҟ?10?A@)5?b??^??1?Eoa??9@:Preprocessing2T
Iterator::Root::ParallelMapV2	PS?????!?b4WG?4@)	PS?????1?b4WG?4@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?k&?ls??!??k +F5@)6??đ?1&30o3@:Preprocessing2E
Iterator::Root}?ݮ????!?v??$6B@)???$????1??/T/@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?\???!???R?$@)?\???1???R?$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip2??n??!"?}H??O@)P?,?cyw?1?Ǵ??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???=?Z?!_'??/q??)???=?Z?1_'??/q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?44.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?D?FY?F@QM?#??zK@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	n??S??n??S??!n??S??      ??!       "	z??9y	,@z??9y	,@!z??9y	,@*      ??!       2      ??!       :	l??+?&@l??+?&@!l??+?&@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?D?FY?F@yM?#??zK@?
"?
_gradient_tape/model/efficientnetb0/block1a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter????>??!????>??0"?
_gradient_tape/model/efficientnetb0/block2b_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?k?fz???!蚉?8??0"?
_gradient_tape/model/efficientnetb0/block2a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter%9??k??!?T|6????0"?
_gradient_tape/model/efficientnetb0/block3a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?X???*??!!jihl??0".
IteratorGetNext/_26_RecvF+`.?{?!9??L???"?
_gradient_tape/model/efficientnetb0/block4a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?]ҏ??s?!?`)?0+??0"s
Igradient_tape/model/efficientnetb0/block2a_expand_bn/FusedBatchNormGradV3FusedBatchNormGradV3T??RP9r?!?rt?t??"q
Fgradient_tape/model/efficientnetb0/top_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?s?Kޠq?!???j????0"~
Rgradient_tape/model/efficientnetb0/block2a_expand_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??CyFXq?!Ĉ?????0"N
0model/efficientnetb0/block6c_project_conv/Conv2DConv2D??!<??o?!?z>??0I_?f!?á?Q#?;???X@Yc7??#???a"W?p;?X@q?\q;?#@y??*?ǐ?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?44.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 