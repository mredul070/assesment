?	?!?aK?3@?!?aK?3@!?!?aK?3@	?Gz<'????Gz<'???!?Gz<'???"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9?!?aK?3@?)??F???1??p???)@A?_>Y1\m?I'P?"??@Y?W?B???r0*	7?A`?@[@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?g??????!?!???A@)q9^??I??1	???j?6@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat4???lɚ?!????7@)0?GĔ??1?%6@:Preprocessing2E
Iterator::Root??G??'??!?
??B@)<?R?!???1Ƨ?7O3@:Preprocessing2T
Iterator::Root::ParallelMapV2?[<?????!?O????0@)?[<?????1?O????0@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?,?cyW??!h"i??H*@)?,?cyW??1h"i??H*@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip1?*?Ա?!7??<?O@)`s?	Mr?1??.??d@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorX??C?a?!@@??A???)X??C?a?1@@??A???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?31.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?Gz<'???I?r?Q?u@@Q?Q?k?P@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?)??F????)??F???!?)??F???      ??!       "	??p???)@??p???)@!??p???)@*      ??!       2	?_>Y1\m??_>Y1\m?!?_>Y1\m?:	'P?"??@'P?"??@!'P?"??@B      ??!       J	?W?B????W?B???!?W?B???R      ??!       Z	?W?B????W?B???!?W?B???b      ??!       JGPUY?Gz<'???b q?r?Q?u@@y?Q?k?P@?"v
Kgradient_tape/model/resnet50/conv5_block1_0_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?;?????!?;?????0"x
Lgradient_tape/model/resnet50/conv5_block2_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterX??L?D??!??af?e??0"x
Lgradient_tape/model/resnet50/conv5_block1_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??@??C??!?#A??C??0"x
Lgradient_tape/model/resnet50/conv5_block3_2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterT??N????!?򶽲??0"I
)model/resnet50/conv5_block1_2_conv/Conv2DConv2D?,F?bŃ?!f???08"I
)model/resnet50/conv5_block3_2_conv/Conv2DConv2Dʈ$x}??!z??7?A??08"I
)model/resnet50/conv5_block2_2_conv/Conv2DConv2D;???z??!a?????08"G
)model/resnet50/conv5_block3_1_conv/Conv2DConv2Dn??AR??!??A?W۶?0"G
)model/resnet50/conv5_block2_1_conv/Conv2DConv2DB????!?*??????0"G
)model/resnet50/conv5_block1_0_conv/Conv2DConv2D'??0??!Iu?????0I???<???Q?E}X??X@Y??F:l???a??On?X@qn??N-?@yʔy{C???"?

device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?31.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Ampere)(: B 