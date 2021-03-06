<!-- Generated by RapidsConf.help. DO NOT EDIT! -->
# Rapids Plugin 4 Spark Configuration
The following is the list of options that `rapids-plugin-4-spark` supports.

On startup use: `--conf [conf key]=[conf value]`. For example:

```
${SPARK_HOME}/bin/spark --jars 'rapids-4-spark_2.12-0.1-SNAPSHOT.jar,cudf-0.14-SNAPSHOT-cuda10.jar' \
--conf spark.plugins=ai.rapids.spark.SQLPlugin \
--conf spark.rapids.sql.incompatibleOps.enabled=true
```

At runtime use: `spark.conf.set("[conf key]", [conf value])`. For example:

```
scala> spark.conf.set("spark.rapids.sql.incompatibleOps.enabled", true)
```

## General Configuration
Name | Description | Default Value
-----|-------------|--------------
spark.rapids.memory.gpu.allocFraction|The fraction of total GPU memory that should be initially allocated for pooled memory. Extra memory will be allocated as needed, but it may result in more fragmentation.|0.9
spark.rapids.memory.gpu.debug|Provides a log of GPU memory allocations and frees. If set to STDOUT or STDERR the logging will go there. Setting it to NONE disables logging. All other values are reserved for possible future expansion and in the mean time will disable logging.|NONE
spark.rapids.memory.gpu.pooling.enabled|Should RMM act as a pooling allocator for GPU memory, or should it just pass through to CUDA memory allocation directly.|true
spark.rapids.memory.gpu.spillAsyncStart|Fraction of device memory utilization at which data will start spilling asynchronously to free up device memory|0.9
spark.rapids.memory.gpu.spillAsyncStop|Fraction of device memory utilization at which data will stop spilling asynchronously to free up device memory|0.8
spark.rapids.memory.host.spillStorageSize|Amount of off-heap host memory to use for buffering spilled GPU data before spilling to local disk|1073741824
spark.rapids.memory.pinnedPool.size|The size of the pinned memory pool in bytes unless otherwise specified. Use 0 to disable the pool.|0
spark.rapids.memory.uvm.enabled|UVM or universal memory can allow main host memory to act essentially as swap for device(GPU) memory. This allows the GPU to process more data than fits in memory, but can result in slower processing. This is an experimental feature.|false
spark.rapids.shuffle.transport.enabled|When set to true, enable the Rapids Shuffle Transport for accelerated shuffle.|false
spark.rapids.shuffle.transport.maxReceiveInflightBytes|Maximum aggregate amount of bytes that be fetched at any given time from peers during shuffle|1073741824
spark.rapids.shuffle.ucx.managementServerHost|The host to be used to start the management server|null
spark.rapids.shuffle.ucx.useWakeup|When set to true, use UCX's event-based progress (epoll) in order to wake up the progress thread when needed, instead of a hot loop.|true
spark.rapids.sql.batchSizeBytes|Set the target number of bytes for a GPU batch. Splits sizes for input data is covered by separate configs.|2147483647
spark.rapids.sql.castFloatToString.enabled|Casting from floating point types to string on the GPU returns results that have a different precision than the default Java toString behavior.|false
spark.rapids.sql.castStringToInteger.enabled|When set to true, enables casting from strings to integer types (byte, short, int, long) on the GPU. Casting from string to integer types on the GPU returns incorrect results when the string represents a number larger than Long.MaxValue or smaller than Long.MinValue.|false
spark.rapids.sql.castTimestampToString.enabled|When set to true, casting from timestamp to string is supported on the GPU. Note that the GPU returns timestamps formatted with trailing zeros for the millisecond part which differs from default Spark behavior|false
spark.rapids.sql.concurrentGpuTasks|Set the number of tasks that can execute concurrently per GPU. Tasks may temporarily block when the number of concurrent tasks in the executor exceeds this amount. Allowing too many concurrent tasks on the same GPU may lead to GPU out of memory errors.|1
spark.rapids.sql.contains.negative.timestamps|Whether the data contains negative timestamps i.e. timestamps prior to Jan 1st 1970. When set to true operators using timestamps will not be accelerated|false
spark.rapids.sql.enabled|Enable (true) or disable (false) sql operations on the GPU|true
spark.rapids.sql.explain|Explain why some parts of a query were not placed on a GPU or not. Possible values are ALL: print everything, NONE: print nothing, NOT_ON_GPU: print only did not go on the GPU|NONE
spark.rapids.sql.hasNans|Config to indicate if your data has NaN's. Cudf doesn't currently support NaN's properly so you can get corrupt data if you have NaN's in your data and it runs on the GPU.|true
spark.rapids.sql.hashOptimizeSort.enabled|Whether sorts should be inserted after some hashed operations to improve output ordering. This can improve output file sizes when saving to columnar formats.|false
spark.rapids.sql.improvedFloatOps.enabled|For some floating point operations spark uses one way to compute the value and the underlying cudf implementation can use an improved algorithm. In some cases this can result in cudf producing an answer when spark overflows. Because this is not as compatible with spark, we have it disabled by default.|false
spark.rapids.sql.incompatibleOps.enabled|For operations that work, but are not 100% compatible with the Spark equivalent set if they should be enabled by default or disabled by default.|false
spark.rapids.sql.reader.batchSizeBytes|Soft limit on the maximum number of bytes the reader reads per batch. The readers will read chunks of data until this limit is met or exceeded. Note that the reader may estimate the number of bytes that will be used on the GPU in some cases based on the schema and number of rows in each batch.|2147483647
spark.rapids.sql.reader.batchSizeRows|Soft limit on the maximum number of rows the reader will read per batch. The orc and parquet readers will read row groups until this limit is met or exceeded. The limit is respected by the csv reader.|2147483647
spark.rapids.sql.replaceSortMergeJoin.enabled|Allow replacing sortMergeJoin with HashJoin|true
spark.rapids.sql.shuffle.spillThreads|Number of threads used to spill shuffle data to disk in the background.|6
spark.rapids.sql.variableFloatAgg.enabled|Spark assumes that all operations produce the exact same result each time. This is not true for some floating point aggregations, which can produce slightly different results on the GPU as the aggregation is done in parallel.  This can enable those operations if you know the query is only computing it once.|false

## Fine Tuning
_Rapids Plugin 4 Spark_ can be further configured to enable or disable specific
expressions and to control what parts of the query execute using the GPU or
the CPU.

Please leverage the `spark.rapids.sql.explain` setting to get feedback from the
plugin as to why parts of a query may not be executing on the GPU.

**NOTE:** Setting `spark.rapids.sql.incompatibleOps.enabled=true` will enable all
the settings in the table below which are not enabled by default due to
incompatibilities.

### Expressions
Name | Description | Default Value | Incompatibilities
-----|-------------|---------------|------------------
spark.rapids.sql.expression.Abs|absolute value|true|None|
spark.rapids.sql.expression.Acos|inverse cosine|true|None|
spark.rapids.sql.expression.Acosh|inverse hyperbolic cosine|true|None|
spark.rapids.sql.expression.Add|addition|true|None|
spark.rapids.sql.expression.Alias|gives a column a name|true|None|
spark.rapids.sql.expression.And|logical and|true|None|
spark.rapids.sql.expression.AnsiCast|convert a column of one type of data into another type|true|None|
spark.rapids.sql.expression.Asin|inverse sine|true|None|
spark.rapids.sql.expression.Asinh|inverse hyperbolic sine|true|None|
spark.rapids.sql.expression.AtLeastNNonNulls|checks if number of non null/Nan values is greater than a given value|true|None|
spark.rapids.sql.expression.Atan|inverse tangent|true|None|
spark.rapids.sql.expression.Atanh|inverse hyperbolic tangent|true|None|
spark.rapids.sql.expression.AttributeReference|references an input column|true|None|
spark.rapids.sql.expression.BitwiseAnd|Returns the bitwise AND of the operands|true|None|
spark.rapids.sql.expression.BitwiseNot|Returns the bitwise NOT of the operands|true|None|
spark.rapids.sql.expression.BitwiseOr|Returns the bitwise OR of the operands|true|None|
spark.rapids.sql.expression.BitwiseXor|Returns the bitwise XOR of the operands|true|None|
spark.rapids.sql.expression.CaseWhen|CASE WHEN expression|true|None|
spark.rapids.sql.expression.Cast|convert a column of one type of data into another type|true|None|
spark.rapids.sql.expression.Cbrt|cube root|true|None|
spark.rapids.sql.expression.Ceil|ceiling of a number|true|None|
spark.rapids.sql.expression.Coalesce|Returns the first non-null argument if exists. Otherwise, null.|true|None|
spark.rapids.sql.expression.Concat|String Concatenate NO separator|true|None|
spark.rapids.sql.expression.Contains|Contains|true|None|
spark.rapids.sql.expression.Cos|cosine|true|None|
spark.rapids.sql.expression.Cosh|hyperbolic cosine|true|None|
spark.rapids.sql.expression.Cot|Returns the cotangent|true|None|
spark.rapids.sql.expression.CurrentRow$|Special boundary for a window frame, indicating stopping at the current row|true|None|
spark.rapids.sql.expression.DateDiff|datediff|true|None|
spark.rapids.sql.expression.DayOfMonth|get the day of the month from a date or timestamp|true|None|
spark.rapids.sql.expression.Divide|division|true|None|
spark.rapids.sql.expression.EndsWith|Ends With|true|None|
spark.rapids.sql.expression.EqualNullSafe|check if the values are equal including nulls <=>|true|None|
spark.rapids.sql.expression.EqualTo|check if the values are equal|true|None|
spark.rapids.sql.expression.Exp|Euler's number e raised to a power|true|None|
spark.rapids.sql.expression.Expm1|Euler's number e raised to a power minus 1|true|None|
spark.rapids.sql.expression.Floor|floor of a number|true|None|
spark.rapids.sql.expression.FromUnixTime|get the String from a unix timestamp|true|None|
spark.rapids.sql.expression.GreaterThan|> operator|true|None|
spark.rapids.sql.expression.GreaterThanOrEqual|>= operator|true|None|
spark.rapids.sql.expression.If|IF expression|true|None|
spark.rapids.sql.expression.In|IN operator|true|None|
spark.rapids.sql.expression.InSet|INSET operator|true|None|
spark.rapids.sql.expression.InitCap|Returns str with the first letter of each word in uppercase. All other letters are in lowercase|true|None|
spark.rapids.sql.expression.InputFileBlockLength|Returns the length of the block being read, or -1 if not available.|true|None|
spark.rapids.sql.expression.InputFileBlockStart|Returns the start offset of the block being read, or -1 if not available.|true|None|
spark.rapids.sql.expression.InputFileName|Returns the name of the file being read, or empty string if not available.|true|None|
spark.rapids.sql.expression.IntegralDivide|division with a integer result|true|None|
spark.rapids.sql.expression.IsNaN|checks if a value is NaN|true|None|
spark.rapids.sql.expression.IsNotNull|checks if a value is not null|true|None|
spark.rapids.sql.expression.IsNull|checks if a value is null|true|None|
spark.rapids.sql.expression.KnownFloatingPointNormalized|tag to prevent redundant normalization|false|This is not 100% compatible with the Spark version because when enabling these, there may be extra groups produced for floating point grouping keys (e.g. -0.0, and 0.0)|
spark.rapids.sql.expression.Length|String Character Length|true|None|
spark.rapids.sql.expression.LessThan|< operator|true|None|
spark.rapids.sql.expression.LessThanOrEqual|<= operator|true|None|
spark.rapids.sql.expression.Like|Like|true|None|
spark.rapids.sql.expression.Literal|holds a static value from the query|true|None|
spark.rapids.sql.expression.Log|natural log|true|None|
spark.rapids.sql.expression.Log10|log base 10|true|None|
spark.rapids.sql.expression.Log1p|natural log 1 + expr|true|None|
spark.rapids.sql.expression.Log2|log base 2|true|None|
spark.rapids.sql.expression.Logarithm|log variable base|true|None|
spark.rapids.sql.expression.Lower|String lowercase operator|false|This is not 100% compatible with the Spark version because in some cases unicode characters change byte width when changing the case. The GPU string conversion does not support these characters. For a full list of unsupported characters see https://github.com/rapidsai/cudf/issues/3132|
spark.rapids.sql.expression.MonotonicallyIncreasingID|Returns monotonically increasing 64-bit integers.|true|None|
spark.rapids.sql.expression.Month|get the month from a date or timestamp|true|None|
spark.rapids.sql.expression.Multiply|multiplication|true|None|
spark.rapids.sql.expression.NaNvl|evaluates to `left` iff left is not NaN, `right` otherwise.|true|None|
spark.rapids.sql.expression.Not|boolean not operator|true|None|
spark.rapids.sql.expression.Or|logical or|true|None|
spark.rapids.sql.expression.Pmod|pmod|true|None|
spark.rapids.sql.expression.Pow|lhs ^ rhs|true|None|
spark.rapids.sql.expression.Rand|Generate a random column with i.i.d. uniformly distributed values in [0, 1)|true|None|
spark.rapids.sql.expression.RegExpReplace|RegExpReplace|true|None|
spark.rapids.sql.expression.Remainder|remainder or modulo|true|None|
spark.rapids.sql.expression.Rint|Rounds up a double value to the nearest double equal to an integer|true|None|
spark.rapids.sql.expression.RowNumber|Window function that returns the index for the row within the aggregation window|true|None|
spark.rapids.sql.expression.ShiftLeft|Bitwise shift left (<<)|true|None|
spark.rapids.sql.expression.ShiftRight|Bitwise shift right (>>)|true|None|
spark.rapids.sql.expression.ShiftRightUnsigned|Bitwise unsigned shift right (>>>)|true|None|
spark.rapids.sql.expression.Signum|Returns -1.0, 0.0 or 1.0 as expr is negative, 0 or positive|true|None|
spark.rapids.sql.expression.Sin|sine|true|None|
spark.rapids.sql.expression.Sinh|hyperbolic sine|true|None|
spark.rapids.sql.expression.SortOrder|sort order|true|None|
spark.rapids.sql.expression.SparkPartitionID|Returns the current partition id.|true|None|
spark.rapids.sql.expression.SpecifiedWindowFrame|specification of the width of the group (or "frame") of input rows around which a window function is evaluated|true|None|
spark.rapids.sql.expression.Sqrt|square root|true|None|
spark.rapids.sql.expression.StartsWith|Starts With|true|None|
spark.rapids.sql.expression.StringLocate|Substring search operator|true|None|
spark.rapids.sql.expression.StringReplace|StringReplace operator|true|None|
spark.rapids.sql.expression.StringTrim|StringTrim operator|true|None|
spark.rapids.sql.expression.StringTrimLeft|StringTrimLeft operator|true|None|
spark.rapids.sql.expression.StringTrimRight|StringTrimRight operator|true|None|
spark.rapids.sql.expression.Substring|Substring operator|true|None|
spark.rapids.sql.expression.SubstringIndex|substring_index operator|true|None|
spark.rapids.sql.expression.Subtract|subtraction|true|None|
spark.rapids.sql.expression.Tan|tangent|true|None|
spark.rapids.sql.expression.Tanh|hyperbolic tangent|true|None|
spark.rapids.sql.expression.TimeSub|Subtracts interval from timestamp|true|None|
spark.rapids.sql.expression.ToDegrees|Converts radians to degrees|true|None|
spark.rapids.sql.expression.ToRadians|Converts degrees to radians|true|None|
spark.rapids.sql.expression.UnaryMinus|negate a numeric value|true|None|
spark.rapids.sql.expression.UnaryPositive|a numeric value with a + in front of it|true|None|
spark.rapids.sql.expression.UnboundedFollowing$|Special boundary for a window frame, indicating all rows preceding the current row|true|None|
spark.rapids.sql.expression.UnboundedPreceding$|Special boundary for a window frame, indicating all rows preceding the current row|true|None|
spark.rapids.sql.expression.UnixTimestamp|convert a string date or timestamp to a unix timestamp|false|This is not 100% compatible with the Spark version because Incorrectly formatted strings and bogus dates produce garbage data instead of null|
spark.rapids.sql.expression.Upper|String uppercase operator|false|This is not 100% compatible with the Spark version because in some cases unicode characters change byte width when changing the case. The GPU string conversion does not support these characters. For a full list of unsupported characters see https://github.com/rapidsai/cudf/issues/3132|
spark.rapids.sql.expression.WindowExpression|calculates a return value for every input row of a table based on a group (or "window") of rows|true|None|
spark.rapids.sql.expression.WindowSpecDefinition|specification of a window function, indicating the partitioning-expression, the row ordering, and the width of the window|true|None|
spark.rapids.sql.expression.Year|get the year from a date or timestamp|true|None|
spark.rapids.sql.expression.AggregateExpression|aggregate expression|true|None|
spark.rapids.sql.expression.Average|average aggregate operator|true|None|
spark.rapids.sql.expression.Count|count aggregate operator|true|None|
spark.rapids.sql.expression.First|first aggregate operator|true|None|
spark.rapids.sql.expression.Last|last aggregate operator|true|None|
spark.rapids.sql.expression.Max|max aggregate operator|true|None|
spark.rapids.sql.expression.Min|min aggregate operator|true|None|
spark.rapids.sql.expression.Sum|sum aggregate operator|true|None|
spark.rapids.sql.expression.NormalizeNaNAndZero|normalize nan and zero|false|This is not 100% compatible with the Spark version because when enabling these, there may be extra groups produced for floating point grouping keys (e.g. -0.0, and 0.0)|

### Execution
Name | Description | Default Value | Incompatibilities
-----|-------------|---------------|------------------
spark.rapids.sql.exec.CoalesceExec|The backend for the dataframe coalesce method|true|None|
spark.rapids.sql.exec.CollectLimitExec|Reduce to single partition and apply limit|true|None|
spark.rapids.sql.exec.ExpandExec|The backend for the expand operator|true|None|
spark.rapids.sql.exec.FileSourceScanExec|Reading data from files, often from Hive tables|true|None|
spark.rapids.sql.exec.FilterExec|The backend for most filter statements|true|None|
spark.rapids.sql.exec.GenerateExec|The backend for operations that generate more output rows than input rows like explode.|true|None|
spark.rapids.sql.exec.GlobalLimitExec|Limiting of results across partitions|true|None|
spark.rapids.sql.exec.LocalLimitExec|Per-partition limiting of results|true|None|
spark.rapids.sql.exec.ProjectExec|The backend for most select, withColumn and dropColumn statements|true|None|
spark.rapids.sql.exec.SortExec|The backend for the sort operator|true|None|
spark.rapids.sql.exec.UnionExec|The backend for the union operator|true|None|
spark.rapids.sql.exec.HashAggregateExec|The backend for hash based aggregations|true|None|
spark.rapids.sql.exec.SortAggregateExec|The backend for sort based aggregations|true|None|
spark.rapids.sql.exec.DataWritingCommandExec|Writing data|true|None|
spark.rapids.sql.exec.BatchScanExec|The backend for most file input|true|None|
spark.rapids.sql.exec.BroadcastExchangeExec|The backend for broadcast exchange of data|true|None|
spark.rapids.sql.exec.ShuffleExchangeExec|The backend for most data being exchanged between processes|true|None|
spark.rapids.sql.exec.BroadcastHashJoinExec|Implementation of join using broadcast data|true|None|
spark.rapids.sql.exec.ShuffledHashJoinExec|Implementation of join using hashed shuffled data|true|None|
spark.rapids.sql.exec.SortMergeJoinExec|Sort merge join, replacing with shuffled hash join|true|None|
spark.rapids.sql.exec.WindowExec|Window-operator backend|true|None|

### Scans
Name | Description | Default Value | Incompatibilities
-----|-------------|---------------|------------------
spark.rapids.sql.input.CSVScan|CSV parsing|true|None|
spark.rapids.sql.input.OrcScan|ORC parsing|true|None|
spark.rapids.sql.input.ParquetScan|Parquet parsing|true|None|

### Partitioning
Name | Description | Default Value | Incompatibilities
-----|-------------|---------------|------------------
spark.rapids.sql.partitioning.HashPartitioning|Hash based partitioning|true|None|
spark.rapids.sql.partitioning.RangePartitioning|Range Partitioning|true|None|
spark.rapids.sql.partitioning.RoundRobinPartitioning|Round Robin Partitioning|true|None|
spark.rapids.sql.partitioning.SinglePartition$|Single Partitioning|true|None|
