?	PS????<@PS????<@!PS????<@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0PS????<@??h?????1?G??[@-@A????}rT?I?\????+@r0*	I+?[@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatT8?T???!?twr?=:@)?$???7??1ș???8@:Preprocessing2T
Iterator::Root::ParallelMapV2?fc%?Y??!????B?6@)?fc%?Y??1????B?6@:Preprocessing2E
Iterator::Root#??]???!*F??a?D@)np?????1??Z??2@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?u7Ouȝ?!?a?=??:@)??v?
???1c?`??*@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice|H??ߠ??!x`?&?*@)|H??ߠ??1x`?&?*@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???-??!׹x?(M@)??	m9w?19s>???@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor6??\^?!X????\??)6??\^?1X????\??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?48.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIt????H@Q??9Y[I@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??h???????h?????!??h?????      ??!       "	?G??[@-@?G??[@-@!?G??[@-@*      ??!       2	????}rT?????}rT?!????}rT?:	?\????+@?\????+@!?\????+@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qt????H@y??9Y[I@?
"?
agradient_tape/model_1/efficientnetb0/block1a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??2ɲ???!??2ɲ???0"?
agradient_tape/model_1/efficientnetb0/block2b_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilteru.?%t|??!?'??W??0"?
agradient_tape/model_1/efficientnetb0/block2a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??ԏ?Ħ?! ?????0"?
agradient_tape/model_1/efficientnetb0/block3a_dwconv/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??9sj??!J?AG????0".
IteratorGetNext/_26_Recv?iD?z?!b??X ??"u
Kgradient_tape/model_1/efficientnetb0/block2a_expand_bn/FusedBatchNormGradV3FusedBatchNormGradV3?a????q?!???l%g??"s
Hgradient_tape/model_1/efficientnetb0/top_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputt??,?{p?!??t%???0"?
Tgradient_tape/model_1/efficientnetb0/block2a_expand_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???B6p?!?/?~????0"~
Sgradient_tape/model_1/efficientnetb0/block6d_expand_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputs????o?!E/??,)??0"~
Sgradient_tape/model_1/efficientnetb0/block6c_expand_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputM??d?o?!߱Ych??0I?.?ߵy??Q:D???X@Y??-????afH??G?X@q??]?HW@y??_#?7??"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?48.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?93.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 