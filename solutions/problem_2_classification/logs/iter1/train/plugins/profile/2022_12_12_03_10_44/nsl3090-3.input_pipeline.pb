	J????9@J????9@!J????9@      ??!       "q
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
	t?//?>??t?//?>??!t?//?>??      ??!       "	s?`?,@s?`?,@!s?`?,@*      ??!       2	6!?1??p?6!?1??p?!6!?1??p?:	?Q??&@?Q??&@!?Q??&@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q	?L???F@y?v?toK@