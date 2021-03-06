/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import java.io.File
import java.nio.charset.StandardCharsets

import collection.JavaConverters._

import org.apache.hadoop.fs.Path
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.util.HadoopInputFile

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.SparkConf

/**
 * Tests for writing Parquet files with the GPU.
 */
class ParquetWriterSuite extends SparkQueryCompareTestSuite {

  def readParquet(spark: SparkSession, path: String): DataFrame = spark.read.parquet(path)

  def writeParquet(df: DataFrame, path: String): Unit = df.write.parquet(path)

  def writeParquetBucket(colNames: String*): (DataFrame, String) => Unit =
    (df, path) => df.write.partitionBy(colNames:_*).parquet(path)

  testSparkWritesAreEqual("simple Parquet write without nulls",
    mixedDf, writeParquet, readParquet)

  testSparkWritesAreEqual("simple Parquet write with nulls",
    mixedDfWithNulls, writeParquet, readParquet)

  testSparkWritesAreEqual("simple partitioned Parquet write",
    mixedDfWithBuckets, writeParquetBucket("bucket_1", "bucket_2"), readParquet,
    sort = true /*The order the data is read in on the CPU is not deterministic*/)

  test("write with no compression") {
    val compression = "none"
    val expectedFileExt = ".parquet"
    val cpuCodecs = withCpuSparkSession(
      spark => getCompressionCodecs(spark, compression, expectedFileExt))
    val gpuCodecs = withGpuSparkSession(
      spark => getCompressionCodecs(spark, compression, expectedFileExt))
    assert(cpuCodecs.forall(_ == "UNCOMPRESSED"))
    assert(gpuCodecs.forall(_ == "UNCOMPRESSED"))
  }

  test("write with snappy compression") {
    val compression = "snappy"
    val expectedFileExt = ".snappy.parquet"
    val cpuCodecs = withCpuSparkSession(
      spark => getCompressionCodecs(spark, compression, expectedFileExt))
    val gpuCodecs = withGpuSparkSession(
      spark => getCompressionCodecs(spark, compression, expectedFileExt))
    assert(cpuCodecs.contains("SNAPPY"))
    assert(gpuCodecs.contains("SNAPPY"))
  }

  test("file metadata") {
    val tempFile = File.createTempFile("stats", ".parquet")
    try {
      withGpuSparkSession(spark => {
        val df = mixedDfWithNulls(spark)
        df.write.mode("overwrite").parquet(tempFile.getAbsolutePath)

        val footer = ParquetFileReader.readFooters(spark.sparkContext.hadoopConfiguration,
          new Path(tempFile.getAbsolutePath)).get(0)

        val parquetMeta = footer.getParquetMetadata
        val fileMeta = footer.getParquetMetadata.getFileMetaData
        val extra = fileMeta.getKeyValueMetaData
        assert(extra.containsKey("org.apache.spark.version"))
        assert(extra.containsKey("org.apache.spark.sql.parquet.row.metadata"))

        val blocks = parquetMeta.getBlocks
        assertResult(1) { blocks.size }
        val block = blocks.get(0)
        assertResult(11) { block.getRowCount }
        val cols = block.getColumns
        assertResult(4) { cols.size }

        assertResult(3) { cols.get(0).getStatistics.getNumNulls }
        assertResult(-700L) { cols.get(0).getStatistics.genericGetMin }
        assertResult(1200L) { cols.get(0).getStatistics.genericGetMax }

        assertResult(4) { cols.get(1).getStatistics.getNumNulls }
        assertResult(1.0) { cols.get(1).getStatistics.genericGetMin }
        assertResult(9.0) { cols.get(1).getStatistics.genericGetMax }

        assertResult(4) { cols.get(2).getStatistics.getNumNulls }
        assertResult(90) { cols.get(2).getStatistics.genericGetMin }
        assertResult(99) { cols.get(2).getStatistics.genericGetMax }

        assertResult(1) { cols.get(3).getStatistics.getNumNulls }
        assertResult("A") {
          new String(cols.get(3).getStatistics.getMinBytes, StandardCharsets.UTF_8)
        }
        assertResult("\ud720\ud721") {
          new String(cols.get(3).getStatistics.getMaxBytes, StandardCharsets.UTF_8)
        }
      })
    } finally {
      tempFile.delete()
    }
  }

  testExpectedExceptionStartsWith(
      "int96 timestamps not supported",
      classOf[IllegalArgumentException],
      "Part of the plan is not columnar",
      frameFromParquet("timestamp-date-test-msec.parquet"),
      new SparkConf().set("spark.sql.parquet.outputTimestampType", "INT96")) {
    val tempFile = File.createTempFile("int96", "parquet")
    tempFile.delete()
    frame => {
      frame.write.parquet(tempFile.getAbsolutePath)
      frame
    }
  }

  private def getCompressionCodecs(
      spark: SparkSession,
      compression: String, expectedExt: String): Seq[String] = {
    val tempFile = File.createTempFile(s"compression-$compression-test", ".parquet")

    try {
      val df = mixedDfWithNulls(spark)
      df.write
        .mode("overwrite")
        .option("compression", compression)
        .parquet(tempFile.getAbsolutePath)

      val files = new File(tempFile.getAbsolutePath)
        .list()
        .filter { filename =>
          val i = filename.indexOf('.')
          i != -1 && filename.substring(i) == expectedExt
        }

      assert(files.length > 0)

      val file = HadoopInputFile.fromPath(new Path(new File(tempFile, files.head).getAbsolutePath),
        spark.sparkContext.hadoopConfiguration)

      val fileReader = ParquetFileReader.open(file)

      fileReader.getRowGroups.asScala
        .flatMap(_.getColumns.asScala
          .map(_.getCodec.toString))

    } finally {
      tempFile.delete()
    }
  }
}
