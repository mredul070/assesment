	PS????<@PS????<@!PS????<@      ??!       "q
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
	??h???????h?????!??h?????      ??!       "	?G??[@-@?G??[@-@!?G??[@-@*      ??!       2	????}rT?????}rT?!????}rT?:	?\????+@?\????+@!?\????+@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qt????H@y??9Y[I@