	,?,?	3@,?,?	3@!,?,?	3@      ??!       "h
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
	???O???????O????!???O????      ??!       "	?I}Y?*@?I}Y?*@!?I}Y?*@*      ??!       2      ??!       :	?Nyt#?@?Nyt#?@!?Nyt#?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??}G???@y͂ n?Q@