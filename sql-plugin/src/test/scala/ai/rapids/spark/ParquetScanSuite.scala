/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.col

class ParquetScanSuite extends SparkQueryCompareTestSuite {

  testSparkResultsAreEqual("Test Parquet", frameFromParquet("test.snappy.parquet")) {
    frame => frame.select(col("ints_1"), col("ints_3"), col("ints_5"))
  }

  private val fileSplitsParquet = frameFromParquet("file-splits.parquet")

  private val parquetSplitsConf = new SparkConf().set("spark.sql.files.maxPartitionBytes", "10000")

  testSparkResultsAreEqual("Test Parquet file splitting", fileSplitsParquet,
    conf=parquetSplitsConf) {
    frame => frame.select(col("*"))
  }

  testSparkResultsAreEqual("Test Parquet with row chunks", fileSplitsParquet,
    conf = new SparkConf().set(RapidsConf.MAX_READER_BATCH_SIZE_ROWS.key, "100")) {
    frame => frame.select(col("*"))
  }

  testSparkResultsAreEqual("Test Parquet with byte chunks", fileSplitsParquet,
    conf = new SparkConf().set(RapidsConf.MAX_READER_BATCH_SIZE_BYTES.key, "100")) {
    frame => frame.select(col("*"))
  }

  testSparkResultsAreEqual("Test Parquet count", fileSplitsParquet,
    conf=parquetSplitsConf)(frameCount)

  testSparkResultsAreEqual("Test Parquet predicate push-down", fileSplitsParquet) {
    frame => frame.select(col("loan_id"), col("orig_interest_rate"), col("zip"))
      .where(col("orig_interest_rate") > 10)
  }

  testSparkResultsAreEqual("Test Parquet splits predicate push-down", fileSplitsParquet,
    conf=parquetSplitsConf) {
    frame => frame.select(col("loan_id"), col("orig_interest_rate"), col("zip"))
      .where(col("orig_interest_rate") > 10)
  }

  testSparkResultsAreEqual("Test partitioned Parquet", frameFromParquet("partitioned-parquet")) {
    frame => frame.select(col("partKey"), col("ints_1"), col("ints_3"), col("ints_5"))
  }

  testSparkResultsAreEqual("Test Parquet msec timestamps and dates",
      frameFromParquet("timestamp-date-test-msec.parquet")) {
    frame => frame.select(col("*"))
  }

  // This test is commented out because cudf doesn't support loading anything more than
  // millsecond resolution timestamps yet.  See https://github.com/rapidsai/cudf/issues/2497
  // NOTE: When this is fixed, the timestamp msec test and data file should be deleted
  //       in preference of this test.
  //testSparkResultsAreEqual("Test Parquet timestamps and dates",
  //  frameFromParquet("timestamp-date-test.parquet")) {
  //  frame => frame.select(col("*"))
  //}
}
