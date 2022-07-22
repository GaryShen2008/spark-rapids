/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
package org.apache.spark.sql.rapids.execution.python

import java.io.DataOutputStream
import java.net.Socket

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.{CudfColumn, GpuColumnVector, GpuMetric}
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.ArrowStreamWriter

import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.api.python.{ChainedPythonFunctions, PythonRDD}
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.execution.arrow.ArrowWriter
import org.apache.spark.sql.execution.python.PythonUDFRunner
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.unsafe.types.UTF8String
import org.apache.spark.util.Utils

/**
 * Similar to `PythonUDFRunner`, but exchange data with Python worker via Arrow stream.
 */
class GpuArrowCudaIcEvalPythonExec(
    funcs: Seq[ChainedPythonFunctions],
    evalType: Int,
    argOffsets: Array[Array[Int]],
    pythonInSchema: StructType,
    timeZoneId: String,
    conf: Map[String, String],
    batchSize: Long,
    semWait: GpuMetric,
    onDataWriteFinished: () => Unit,
    pythonOutSchema: StructType,
    minReadTargetBatchSize: Int = 1)
  extends GpuArrowPythonRunner(funcs, evalType, argOffsets, pythonInSchema,
    timeZoneId, conf, batchSize, semWait, onDataWriteFinished,
    pythonOutSchema, minReadTargetBatchSize) {

//  override val bufferSize: Int = 65536
//  require(
//    bufferSize >= 4,
//    "Pandas execution requires more than 4 bytes. Please set higher buffer. " +
//      s"Please change '${SQLConf.PANDAS_UDF_BUFFER_SIZE.key}'.")

  protected override def newWriterThread(
    env: SparkEnv,
    worker: Socket,
    inputIterator: Iterator[ColumnarBatch],
    partitionIndex: Int,
    context: TaskContext): WriterThread = {
    new WriterThread(env, worker, inputIterator, partitionIndex, context) {

      protected override def writeCommand(dataOut: DataOutputStream): Unit = {

        // Write config for the worker as a number of key -> value pairs of strings
        dataOut.writeInt(conf.size)
        for ((k, v) <- conf) {
          PythonRDD.writeUTF(k, dataOut)
          PythonRDD.writeUTF(v, dataOut)
        }

        PythonUDFRunner.writeUDFs(dataOut, funcs, argOffsets)
      }

      private def getTableIpcInfo(table: Table): Array[Any] = {
        val columnIpcInfo = new Array[Any](table.getNumberOfColumns)
        for (i <- 0 until table.getNumberOfColumns) {
          columnIpcInfo(i) = CudfColumn.from(table.getColumn(i)).getArrayInterfaceJson
        }
        columnIpcInfo
      }

      protected override def writeIteratorToStream(dataOut: DataOutputStream): Unit = {
        // create schema with all string types
        // TODO, we don't support nested types
        val stringSchema = StructType(pythonInSchema.map(f =>
          StructField(f.name, StringType, f.nullable)))

        val arrowSchema = ArrowUtils.toArrowSchema(stringSchema, timeZoneId)
        val allocator = ArrowUtils.rootAllocator.newChildAllocator(
          s"stdout writer for $pythonExec", 0, Long.MaxValue)
        val root = VectorSchemaRoot.create(arrowSchema, allocator)

        Utils.tryWithSafeFinally {
          val arrowWriter = ArrowWriter.create(root)
          val writer = new ArrowStreamWriter(root, null, dataOut)
          writer.start()

          while (inputIterator.hasNext) {
            val table = withResource(inputIterator.next()) { nextBatch =>
              GpuColumnVector.from(nextBatch)
            }
            val columnIpcInfo: Array[Any] = getTableIpcInfo(table)
            val finalIpcs: Array[Any] = columnIpcInfo.map(col =>
              UTF8String.fromString(col.toString))
            val genericInternalRow = new GenericInternalRow(finalIpcs)
            arrowWriter.write(genericInternalRow)
            arrowWriter.finish()
            writer.writeBatch()
            arrowWriter.reset()
          }
          // end writes footer to the output stream and doesn't clean any resources.
          // It could throw exception if the output stream is closed, so it should be
          // in the try block.
          writer.end()
        } {
          // If we close root and allocator in TaskCompletionListener, there could be a race
          // condition where the writer thread keeps writing to the VectorSchemaRoot while
          // it's being closed by the TaskCompletion listener.
          // Closing root and allocator here is cleaner because root and allocator is owned
          // by the writer thread and is only visible to the writer thread.
          //
          // If the writer thread is interrupted by TaskCompletionListener, it should either
          // (1) in the try block, in which case it will get an InterruptedException when
          // performing io, and goes into the finally block or (2) in the finally block,
          // in which case it will ignore the interruption and close the resources.
          root.close()
          allocator.close()
        }
      }
    }
  }

}
