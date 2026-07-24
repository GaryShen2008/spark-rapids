/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

/*** spark-rapids-shim-json-lines
{"spark": "330"}
spark-rapids-shim-json-lines ***/
package org.apache.spark.sql.rapids.suites

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.datasources.FileSourceCodecSuite
import org.apache.spark.sql.execution.datasources.OrcCodecSuite
import org.apache.spark.sql.execution.datasources.ParquetCodecSuite
import org.apache.spark.sql.rapids.{ExecutionPlanCaptureCallback, GpuFileSourceScanExec}
import org.apache.spark.sql.rapids.utils.RapidsSQLTestsTrait

trait RapidsFileSourceCodecGpuChecks extends FileSourceCodecSuite with RapidsSQLTestsTrait {
  protected def unsupportedGpuCodecs: Set[String] = Set.empty

  private def requireGpuWrite(plans: Array[SparkPlan], codec: String): Unit = {
    assert(
      plans.exists(ExecutionPlanCaptureCallback.contains(_, "GpuDataWritingCommandExec")),
      s"Expected GpuDataWritingCommandExec for $format codec $codec, found:\n" +
        plans.map(_.treeString).mkString("\n\n"))
  }

  private def requireGpuRead(df: DataFrame, codec: String): Unit = {
    val plan = df.queryExecution.executedPlan
    val gpuScans = plan.collect { case scan: GpuFileSourceScanExec => scan }
    assert(
      gpuScans.nonEmpty,
      s"Expected GpuFileSourceScanExec for $format codec $codec, found:\n$plan")
  }

  availableCodecs.filterNot(unsupportedGpuCodecs).foreach { codec =>
    testRapids(s"write and read - file source $format - codec: $codec") {
      withSQLConf(codecConfigName -> codec) {
        withTempPath { dir =>
          ExecutionPlanCaptureCallback.startCapture()
          testData.repartition(5).write.format(format).save(dir.getCanonicalPath)
          requireGpuWrite(ExecutionPlanCaptureCallback.getResultsWithTimeout(), codec)

          val df = spark.read.format(format).load(dir.getCanonicalPath)
          requireGpuRead(df, codec)
          checkAnswer(df, testData)
        }
      }
    }
  }
}

class RapidsParquetCodecSuite extends ParquetCodecSuite with RapidsFileSourceCodecGpuChecks {
  override protected def unsupportedGpuCodecs: Set[String] = Set("gzip", "lz4")
}

class RapidsOrcCodecSuite extends OrcCodecSuite with RapidsFileSourceCodecGpuChecks {
  override protected def unsupportedGpuCodecs: Set[String] = Set("lz4", "lzo")
}
