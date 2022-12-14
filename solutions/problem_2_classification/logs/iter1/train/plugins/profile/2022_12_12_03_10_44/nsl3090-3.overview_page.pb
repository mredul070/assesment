?	J????9@J????9@!J????9@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0J????9@t?//?>??1s?`?,@A6!?1??p?I?Q??&@r0*	X9???Z@2T
Iterator::Root::ParallelMapV2???
?H??!?Jp<d?9@)???
?H??1?Jp<d?9@:Preprocessing2E
Iterator::Root?Y-??D??!,?Q?G@)???U?@??1V??>6@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?rJ_??!?ʢ?)?6@)???-???1ۈM=?4@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?ypw?n??!???-??8@)??Ҥt??1no1?? )@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceAJ?i??!M$????(@)AJ?i??1M$????(@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???3???!???t?J@)ADj??4s?1?L???}@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorX?%???c?!?}πc?@)X?%???c?1?}πc?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?44.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI	?L???F@Q?v?toK@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	t?//?>??t?//?>??!t?//?>??      ??!       "	s?`?,@s?`?,@!s?`?,@*      ??!       2	6!?1??p?6!?1??p?!6!?1??p?:	?Q??&@?Q??&@!?Q??&@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q	?L???F@y?v?toK@?
"?
agradient_tape/model_1/efficientnetb0/block1a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??+~?>??!??+~?>??0"?
agradient_tape/model_1/efficientnetb0/block2b_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterxe?ͤ??!?h??????0"?
agradient_tape/model_1/efficientnetb0/block2a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??????!pGve}t??0"?
agradient_tape/model_1/efficientnetb0/block3a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?g%kΚ?!?=?d!??0".
IteratorGetNext/_26_Recv*?RƩ?{?!?羦???"?
agradient_tape/model_1/efficientnetb0/block4a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterh?'|9u?!%(??????0"u
Kgradient_tape/model_1/efficientnetb0/block2a_expand_bn/FusedBatchNormGradV3FusedBatchNormGradV3?KR?:Yr?!Sq??-??"?
Tgradient_tape/model_1/efficientnetb0/block2a_expand_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?=V?0Rq?!Jʎ?8s??0"s
Hgradient_tape/model_1/efficientnetb0/top_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputX=??Jq?!?#?kd???0"P
2model_1/efficientnetb0/block6c_project_conv/Conv2DConv2D?Gt?o?!?&\T???0IK?T? ??Qo?f???X@Y ??G????a\??X@q?????W@y?? ?????"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?44.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?94.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 