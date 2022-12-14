?	,?,?	3@,?,?	3@!,?,?	3@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails',?,?	3@???O????1?I}Y?*@I?Nyt#?@r0*	??? ??V@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????8+??!f?H?^TC@)d???_???1y??0?$8@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?Fw;S??!4?Ż?9@):??)??1?V?*}?7@:Preprocessing2T
Iterator::Root::ParallelMapV2?p>????!?????1@)?p>????1?????1@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???9#J??!????t-@)???9#J??1????t-@:Preprocessing2E
Iterator::Root" 8?친?!7 ????>@)?ip[[??1???)@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??z?2Q??!?w??\Q@)~?[?~lr?1 ?nЙ@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoruʣaQa?!?ZB??l@)uʣaQa?1?ZB??l@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?30.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??}G???@Q͂ n?Q@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???O???????O????!???O????      ??!       "	?I}Y?*@?I}Y?*@!?I}Y?*@*      ??!       2      ??!       :	?Nyt#?@?Nyt#?@!?Nyt#?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??}G???@y͂ n?Q@?"v
Kgradient_tape/model/resnet50/conv5_block1_0_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput??`??6??!??`??6??0"x
Lgradient_tape/model/resnet50/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter <?|?*??!Z???0??0"x
Lgradient_tape/model/resnet50/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!???????0"x
Lgradient_tape/model/resnet50/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterP??????!??g:???0"I
)model/resnet50/conv5_block1_2_conv/Conv2DConv2D??(?˃?!??`9???08"I
)model/resnet50/conv5_block2_2_conv/Conv2DConv2D?A6d:i??!?L??C.??08"I
)model/resnet50/conv5_block3_2_conv/Conv2DConv2Db???P??!7?13U???08"G
)model/resnet50/conv5_block3_1_conv/Conv2DConv2D[?????!թ?ʹ??0"G
)model/resnet50/conv5_block2_1_conv/Conv2DConv2DL?a???!?>>??׸?0"G
)model/resnet50/conv5_block1_0_conv/Conv2DConv2D9EV?>?!٢c??Ⱥ?0I???ۣ?Q-c???X@Y??F:l???a??On?X@q???V??$@yi?\
J??"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?30.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 