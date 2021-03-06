/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.spark

import ai.rapids.cudf.{ColumnVector, Scalar}

import org.apache.spark.sql.types.{DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.TaskContext

/**
 * An expression that returns the current partition id just like
 * [[org.apache.spark.sql.catalyst.expressions.SparkPartitionID]]
 */
case class GpuSparkPartitionID() extends GpuLeafExpression {
  /**
   * We need to recompute this if something fails.
   */
  override lazy val deterministic: Boolean = false
  override def foldable: Boolean = false
  override def nullable: Boolean = false
  override def dataType: DataType = IntegerType

  @transient private[this] var partitionId: Int = _
  @transient private[this] var wasInitialized: Boolean = _

  override val prettyName = "SPARK_PARTITION_ID"

  override def columnarEval(batch: ColumnarBatch): Any = {
    if (!wasInitialized) {
      partitionId = TaskContext.getPartitionId()
      wasInitialized = true
    }
    var part: Scalar = null
    try {
      val numRows = batch.numRows()
      part = Scalar.fromInt(partitionId)
      GpuColumnVector.from(ColumnVector.fromScalar(part, numRows))
    } finally {
      if (part != null) {
        part.close()
      }
    }
  }
}
