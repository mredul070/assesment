	b?1???9@b?1???9@!b?1???9@      ??!       "h
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
	n??S??n??S??!n??S??      ??!       "	z??9y	,@z??9y	,@!z??9y	,@*      ??!       2      ??!       :	l??+?&@l??+?&@!l??+?&@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?D?FY?F@yM?#??zK@