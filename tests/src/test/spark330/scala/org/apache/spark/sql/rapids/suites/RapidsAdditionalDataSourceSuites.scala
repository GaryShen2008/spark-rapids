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

import java.io.File
import java.net.URI

import scala.util.Random

import com.nvidia.spark.rapids.GpuShuffledSymmetricHashJoinExec
import org.apache.hadoop.fs.Path

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.execution.DataSourceScanExecRedactionSuite
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.functions.{lit, when}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuFileSourceScanExec
import org.apache.spark.sql.rapids.execution.GpuShuffleExchangeExecBase
import org.apache.spark.sql.rapids.utils.RapidsSQLTestsTrait
import org.apache.spark.sql.sources.BucketedReadWithoutHiveSupportSuite
import org.apache.spark.util.Utils

class RapidsBucketedReadWithoutHiveSupportSuite
  extends BucketedReadWithoutHiveSupportSuite with RapidsSQLTestsTrait {
  import testImplicits._

  private val maxI = 5
  private val maxJ = 13
  private lazy val bucketDf =
    (0 until 50).map(i => (i % maxI, i % maxJ, i.toString)).toDF("i", "j", "k")
  private lazy val nullBucketDf = (for {
    i <- 0 to 50
    s <- Seq(null, "a", "b", "c", "d", "e", "f", null, "g")
  } yield (i % maxI, s, i % maxJ)).toDF("i", "j", "k")
  private lazy val joinDf1 =
    (0 until 50).map(i => (i % 5, i % 13, i.toString)).toDF("i", "j", "k")
  private lazy val joinDf2 =
    (0 until 50).map(i => (i % 7, i % 11, i.toString)).toDF("i", "j", "k")

  private def gpuScans(plan: SparkPlan): Seq[GpuFileSourceScanExec] = {
    collect(plan) { case scan: GpuFileSourceScanExec => scan }
  }

  private def requireGpuScan(df: DataFrame): GpuFileSourceScanExec = {
    gpuScans(df.queryExecution.executedPlan).headOption.getOrElse(
      fail(s"Expected GpuFileSourceScanExec in plan:\n${df.queryExecution.executedPlan}"))
  }

  private def checkGpuBucketRead(
      readDf: DataFrame,
      expectedDf: DataFrame,
      expectedSelectedBuckets: Option[String]): Unit = {
    val sortedRead = readDf.select("i", "j", "k").orderBy("i", "j", "k")
    val sortedExpected = expectedDf.select("i", "j", "k").orderBy("i", "j", "k")
    checkAnswer(sortedRead, sortedExpected)

    val scan = requireGpuScan(sortedRead)
    assert(scan.supportsColumnar)
    expectedSelectedBuckets.foreach { expected =>
      assert(scan.bucketedScan, s"Expected bucketed GPU scan but found:\n$scan")
      assert(scan.metadata("SelectedBucketsCount") === expected)
    }
  }

  private def checkInvalidBucketFile(error: Throwable): Unit = {
    val messages = Iterator.iterate(error)(_.getCause)
      .takeWhile(_ != null)
      .map(_.toString)
      .mkString("\n")
    assert(messages.contains("Invalid bucket file"), messages)
  }

  private def writeBucketedParquet(
      df: DataFrame,
      table: String,
      numBuckets: Int,
      sortByBucketColumns: Boolean = false): Unit = {
    val writer = df.repartition(1).write.format("parquet").bucketBy(numBuckets, "i", "j")
    if (sortByBucketColumns) {
      writer.sortBy("i", "j").saveAsTable(table)
    } else {
      writer.saveAsTable(table)
    }
  }

  private def joinBucketedTables(table1: String, table2: String): DataFrame = {
    val left = spark.table(table1).as("left")
    val right = spark.table(table2).as("right")
    left.join(right, left("i") === right("i") && left("j") === right("j"))
      .select(
        left("i").as("left_i"),
        left("j").as("left_j"),
        left("k").as("left_k"),
        right("i").as("right_i"),
        right("j").as("right_j"),
        right("k").as("right_k"))
  }

  private def expectedBucketedJoin: DataFrame = {
    val left = joinDf1.as("left")
    val right = joinDf2.as("right")
    left.join(right, left("i") === right("i") && left("j") === right("j"))
      .select(
        left("i").as("left_i"),
        left("j").as("left_j"),
        left("k").as("left_k"),
        right("i").as("right_i"),
        right("j").as("right_j"),
        right("k").as("right_k"))
  }

  private def assertGpuJoinAndShuffles(df: DataFrame, expectedShuffles: Int): Unit = {
    val plan = df.queryExecution.executedPlan
    val joins = collect(plan) { case join: GpuShuffledSymmetricHashJoinExec => join }
    assert(joins.nonEmpty, s"Expected GpuShuffledSymmetricHashJoinExec in plan:\n$plan")
    val shuffles = collect(plan) { case shuffle: GpuShuffleExchangeExecBase => shuffle }
    assert(shuffles.length === expectedShuffles, s"Unexpected GPU shuffles in plan:\n$plan")
  }

  private def checkGpuBucketedJoin(joined: DataFrame, expectedShuffles: Int): Unit = {
    checkAnswer(
      joined.orderBy("left_k", "right_k"),
      expectedBucketedJoin.orderBy("left_k", "right_k"))
    assertGpuJoinAndShuffles(joined, expectedShuffles)
  }

  testRapids("read bucketed data") {
    withSQLConf(SQLConf.AUTO_BUCKETED_SCAN_ENABLED.key -> "false") {
      withTable("bucketed_table") {
        bucketDf.write
          .format("parquet")
          .partitionBy("i")
          .bucketBy(8, "j", "k")
          .saveAsTable("bucketed_table")

        val bucketValue = 2
        checkGpuBucketRead(
          spark.table("bucketed_table").filter($"i" === bucketValue),
          bucketDf.filter($"i" === bucketValue),
          Some("8 out of 8"))
      }
    }
  }

  testRapids("read partitioning bucketed tables with bucket pruning filters") {
    withSQLConf(
      SQLConf.AUTO_BUCKETED_SCAN_ENABLED.key -> "false",
      SQLConf.JSON_FILTER_PUSHDOWN_ENABLED.key -> "false") {
      withTable("bucketed_table") {
        bucketDf.write
          .format("json")
          .partitionBy("i")
          .bucketBy(7, "j")
          .saveAsTable("bucketed_table")
        val bucketValue = 5
        checkGpuBucketRead(
          spark.table("bucketed_table").filter($"j" === bucketValue),
          bucketDf.filter($"j" === bucketValue),
          Some("1 out of 7"))
      }
    }
  }

  testRapids("read non-partitioning bucketed tables with bucket pruning filters") {
    withSQLConf(
      SQLConf.AUTO_BUCKETED_SCAN_ENABLED.key -> "false",
      SQLConf.JSON_FILTER_PUSHDOWN_ENABLED.key -> "false") {
      withTable("bucketed_table") {
        bucketDf.write.format("json").bucketBy(7, "j").saveAsTable("bucketed_table")
        val bucketValue = 5
        checkGpuBucketRead(
          spark.table("bucketed_table").filter($"j" === bucketValue),
          bucketDf.filter($"j" === bucketValue),
          Some("1 out of 7"))
      }
    }
  }

  testRapids("read partitioning bucketed tables having null in bucketing key") {
    withSQLConf(
      SQLConf.AUTO_BUCKETED_SCAN_ENABLED.key -> "false",
      SQLConf.JSON_FILTER_PUSHDOWN_ENABLED.key -> "false") {
      withTable("bucketed_table") {
        nullBucketDf.write
          .format("json")
          .partitionBy("i")
          .bucketBy(5, "j")
          .saveAsTable("bucketed_table")
        checkGpuBucketRead(
          spark.table("bucketed_table").filter($"j".isNull),
          nullBucketDf.filter($"j".isNull),
          Some("1 out of 5"))
      }
    }
  }

  testRapids("bucket pruning support IsNaN") {
    withSQLConf(
      SQLConf.AUTO_BUCKETED_SCAN_ENABLED.key -> "false",
      SQLConf.JSON_FILTER_PUSHDOWN_ENABLED.key -> "false") {
      withTable("bucketed_table") {
        val nanDf = nullBucketDf
          .select(
            $"i",
            when($"j".isNull, lit(Double.NaN)).otherwise($"j".cast("double")).as("j"),
            $"k")
        nanDf.write.format("json").bucketBy(5, "j").saveAsTable("bucketed_table")
        checkGpuBucketRead(
          spark.table("bucketed_table").filter($"j".isNaN),
          nanDf.filter($"j".isNaN),
          Some("1 out of 5"))
      }
    }
  }

  testRapids("read partitioning bucketed tables having composite filters") {
    withSQLConf(
      SQLConf.AUTO_BUCKETED_SCAN_ENABLED.key -> "false",
      SQLConf.JSON_FILTER_PUSHDOWN_ENABLED.key -> "false") {
      withTable("bucketed_table") {
        bucketDf.write
          .format("json")
          .partitionBy("i")
          .bucketBy(7, "j")
          .saveAsTable("bucketed_table")
        val bucketValue = 5
        checkGpuBucketRead(
          spark.table("bucketed_table").filter($"j" === bucketValue && $"k" > $"j"),
          bucketDf.filter($"j" === bucketValue && $"k" > $"j"),
          Some("1 out of 7"))
      }
    }
  }

  testRapids("read bucketed table without filters") {
    withTable("bucketed_table") {
      bucketDf.write.format("json").bucketBy(7, "j").saveAsTable("bucketed_table")
      checkGpuBucketRead(spark.table("bucketed_table"), bucketDf, None)
    }
  }

  testRapids("error if there exists any malformed bucket files") {
    withTable("bucketed_table") {
      joinDf1.write.format("parquet").bucketBy(8, "i").saveAsTable("bucketed_table")
      val warehouseFilePath = new URI(spark.sessionState.conf.warehousePath).getPath
      val tableDir = new File(warehouseFilePath, "bucketed_table")
      Utils.deleteRecursively(tableDir)
      joinDf1.write.parquet(tableDir.getAbsolutePath)

      val error = intercept[Exception] {
        spark.table("bucketed_table").groupBy("i").count().count()
      }
      checkInvalidBucketFile(error)
    }
  }

  testRapids("disable bucketing when the output doesn't contain all bucketing columns") {
    withTable("bucketed_table") {
      val numericDf =
        (0 until 50).map(i => (i % 5, i % 13, i)).toDF("i", "j", "k")
      numericDf.write.format("parquet").bucketBy(8, "i").saveAsTable("bucketed_table")

      val scanDf = spark.table("bucketed_table").select("j")
      val scan = requireGpuScan(scanDf)
      assert(!scan.bucketedScan)
      assert(scan.metadata("Bucketed").contains("bucket column(s) not read"))
      checkAnswer(scanDf, numericDf.select("j"))
    }
  }

  testRapids("avoid shuffle when join 2 bucketed tables") {
    withSQLConf(
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "0",
      SQLConf.WHOLESTAGE_CODEGEN_ENABLED.key -> "false",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {
      withTable("bucketed_table1", "bucketed_table2") {
        writeBucketedParquet(joinDf1, "bucketed_table1", 8)
        writeBucketedParquet(joinDf2, "bucketed_table2", 8)
        val joined = joinBucketedTables("bucketed_table1", "bucketed_table2")

        checkGpuBucketedJoin(joined, expectedShuffles = 0)
        val scans = gpuScans(joined.queryExecution.executedPlan)
        assert(scans.length >= 2, "Expected both sides to use GPU scans")
        assert(scans.forall(_.bucketedScan), "Expected bucketed GPU scans")
      }
    }
  }

  testRapids("only shuffle one side when join bucketed table and non-bucketed table") {
    withSQLConf(
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "0",
      SQLConf.WHOLESTAGE_CODEGEN_ENABLED.key -> "false",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {
      withTable("bucketed_table1", "bucketed_table2") {
        writeBucketedParquet(joinDf1, "bucketed_table1", 8)
        joinDf2.write.format("parquet").saveAsTable("bucketed_table2")
        val joined = joinBucketedTables("bucketed_table1", "bucketed_table2")

        checkGpuBucketedJoin(joined, expectedShuffles = 1)
        val scans = gpuScans(joined.queryExecution.executedPlan)
        assert(scans.exists(_.bucketedScan), "Expected bucketed side to use bucketed GPU scan")
        assert(scans.exists(!_.bucketedScan), "Expected non-bucketed side to use GPU scan")
      }
    }
  }

  testRapids("SPARK-29655 Read bucketed tables obeys spark.sql.shuffle.partitions") {
    withSQLConf(
      SQLConf.SHUFFLE_PARTITIONS.key -> "5",
      SQLConf.COALESCE_PARTITIONS_INITIAL_PARTITION_NUM.key -> "7",
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "0",
      SQLConf.WHOLESTAGE_CODEGEN_ENABLED.key -> "false",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {
      withTable("bucketed_table1", "bucketed_table2") {
        writeBucketedParquet(joinDf1, "bucketed_table1", 6)
        joinDf2.write.format("parquet").saveAsTable("bucketed_table2")
        val joined = joinBucketedTables("bucketed_table1", "bucketed_table2")
        checkGpuBucketedJoin(joined, expectedShuffles = 1)

        val scans = gpuScans(joined.queryExecution.executedPlan)
        assert(scans.exists(_.metadata.get("SelectedBucketsCount").contains("6 out of 6")))
      }
    }
  }

  testRapids(
    "SPARK-32767 Bucket join should work if SHUFFLE_PARTITIONS larger than bucket number") {
    withSQLConf(
      SQLConf.SHUFFLE_PARTITIONS.key -> "9",
      SQLConf.COALESCE_PARTITIONS_INITIAL_PARTITION_NUM.key -> "10",
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "0",
      SQLConf.WHOLESTAGE_CODEGEN_ENABLED.key -> "false",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {
      withTable("bucketed_table1", "bucketed_table2") {
        writeBucketedParquet(joinDf1, "bucketed_table1", 8, sortByBucketColumns = true)
        writeBucketedParquet(joinDf2, "bucketed_table2", 6, sortByBucketColumns = true)
        checkGpuBucketedJoin(joinBucketedTables("bucketed_table1", "bucketed_table2"), 1)
      }
    }
  }

  testRapids("bucket coalescing eliminates shuffle") {
    withSQLConf(
      SQLConf.COALESCE_BUCKETS_IN_JOIN_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "0",
      SQLConf.WHOLESTAGE_CODEGEN_ENABLED.key -> "false") {
      withTable("bucketed_table1", "bucketed_table2") {
        writeBucketedParquet(joinDf1, "bucketed_table1", 8, sortByBucketColumns = true)
        writeBucketedParquet(joinDf2, "bucketed_table2", 4, sortByBucketColumns = true)
        checkGpuBucketedJoin(joinBucketedTables("bucketed_table1", "bucketed_table2"), 0)
      }
    }
  }

  testRapids(
    "bucket coalescing is applied when join expressions match with partitioning expressions") {
    withSQLConf(
      SQLConf.COALESCE_BUCKETS_IN_JOIN_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "0",
      SQLConf.WHOLESTAGE_CODEGEN_ENABLED.key -> "false") {
      withTable("t1", "t2") {
        writeBucketedParquet(joinDf1, "t1", 8, sortByBucketColumns = true)
        writeBucketedParquet(joinDf2, "t2", 4, sortByBucketColumns = true)

        def verify(
            query: String,
            expectedNumShuffles: Int,
            expectedCoalescedNumBuckets: Option[Int]): Unit = {
          val df = sql(query)
          df.collect()
          val plan = df.queryExecution.executedPlan
          val shuffles = collect(plan) { case shuffle: GpuShuffleExchangeExecBase => shuffle }
          assert(shuffles.length === expectedNumShuffles, s"Unexpected GPU shuffles:\n$plan")

          val coalescedScans = gpuScans(plan).filter(_.optionalNumCoalescedBuckets.isDefined)
          expectedCoalescedNumBuckets match {
            case Some(expected) =>
              assert(coalescedScans.length === 1, s"Expected one coalesced GPU scan:\n$plan")
              assert(coalescedScans.head.optionalNumCoalescedBuckets.contains(expected))
            case None =>
              assert(coalescedScans.isEmpty, s"Expected no coalesced GPU scan:\n$plan")
          }
        }

        verify("SELECT * FROM t1 JOIN t2 ON t1.i = t2.i AND t1.j = t2.j", 0, Some(4))
        verify(
          "SELECT * FROM t1 JOIN (SELECT i AS x, j AS y FROM t2) " +
            "ON t1.i = x AND t1.j = y",
          0,
          Some(4))
        verify("SELECT * FROM t1 JOIN t2 ON t1.i = t2.i", 2, None)
      }
    }
  }
}

class RapidsDataSourceScanExecRedactionSuite
  extends DataSourceScanExecRedactionSuite with RapidsSQLTestsTrait {

  testRapids("treeString is redacted") {
    withTempDir { dir =>
      val basePath = dir.getCanonicalPath
      spark.range(0, 10).toDF("a").write.orc(new Path(basePath, "foo=1").toString)
      val df = spark.read.orc(basePath)
      val replacement = "*********"

      assert(df.queryExecution.executedPlan.treeString(verbose = true).contains(replacement))
      assert(!df.queryExecution.executedPlan.treeString(verbose = true).contains("file:/"))
      assert(isIncluded(df.queryExecution, replacement))
      assert(isIncluded(df.queryExecution, "GpuFileGpuScan"))
    }
  }

  testRapids("explain is redacted using SQLConf") {
    withTempDir { dir =>
      val basePath = dir.getCanonicalPath
      spark.range(0, 10).toDF("a").write.orc(new Path(basePath, "foo=1").toString)
      val df = spark.read.orc(basePath)
      val replacement = "*********"

      assert(isIncluded(df.queryExecution, replacement))
      assert(isIncluded(df.queryExecution, "GpuFileGpuScan"))
      assert(!isIncluded(df.queryExecution, "file:/"))

      withSQLConf(SQLConf.SQL_STRING_REDACTION_PATTERN.key -> "(?i)GpuFileGpuScan") {
        assert(isIncluded(df.queryExecution, replacement))
        assert(!isIncluded(df.queryExecution, "GpuFileGpuScan"))
        assert(isIncluded(df.queryExecution, "file:/"))
      }
    }
  }

  testRapids("FileSourceScanExec metadata") {
    withTempPath { path =>
      val dir = path.getCanonicalPath
      spark.range(0, 10).write.orc(dir)
      val df = spark.read.orc(dir)
      val plan = df.queryExecution.executedPlan
      val scan = plan.collect { case f: GpuFileSourceScanExec => f }.headOption

      assert(scan.isDefined, s"Expected GpuFileSourceScanExec in plan:\n$plan")
      Seq(
        "Format",
        "ReadSchema",
        "Batched",
        "PartitionFilters",
        "PushedFilters",
        "DataFilters",
        "Location"
      ).foreach { key =>
        assert(scan.get.metadata.contains(key), s"Missing $key in ${scan.get.metadata}")
      }
    }
  }

  testRapids("SPARK-31793: FileSourceScanExec metadata should contain limited file paths") {
    withTempPath { path =>
      val dataDirName = Random.alphanumeric.take(100).toList.mkString
      val dataDir = new File(path, dataDirName)
      dataDir.mkdir()

      val partitionCol = "partitionCol"
      spark.range(10)
        .select("id", "id")
        .toDF("value", partitionCol)
        .write
        .partitionBy(partitionCol)
        .orc(dataDir.getCanonicalPath)
      val paths = (0 to 9).map(i => new File(dataDir, s"$partitionCol=$i").getCanonicalPath)
      val plan = spark.read.orc(paths: _*).queryExecution.executedPlan
      val location = plan collectFirst {
        case f: GpuFileSourceScanExec => f.metadata("Location")
      }

      assert(location.isDefined, s"Expected GpuFileSourceScanExec in plan:\n$plan")
      assert(location.get.contains(paths.head))
      assert(location.get.contains("["))
      assert(location.get.contains("]"))

      val pathsInLocation = location.get.substring(
        location.get.indexOf("[") + 1, location.get.indexOf("]")).split(", ").toSeq
      assert(pathsInLocation.size < paths.size)
    }
  }
}
