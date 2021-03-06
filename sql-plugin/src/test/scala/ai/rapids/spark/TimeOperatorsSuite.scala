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

import java.sql.Date

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._

class TimeOperatorsSuite extends SparkQueryCompareTestSuite {
  testSparkResultsAreEqual("Test datediff", datesDf) {
    frame => frame.select(datediff(col("dates"), col("more_dates")))
  }

  testSparkResultsAreEqual("Test datediff rhs literal", datesDf) {
    frame => frame.select(datediff(col("dates"), lit(Date.valueOf("2020-01-09"))))
  }

  testSparkResultsAreEqual("Test datediff lhs literal", datesDf) {
    frame => frame.select(datediff(lit(Date.valueOf("2018-02-12")), col("more_dates")))
  }

  testSparkResultsAreEqual("Test from_unixtime", datesPostEpochDf) {
    frame => frame.select(from_unixtime(col("dates")))
  }

  testSparkResultsAreEqual("Test from_unixtime with pattern dd/mm/yyyy", datesPostEpochDf) {
    frame => frame.select(from_unixtime(col("dates"),"dd/MM/yyyy"))
  }

  testSparkResultsAreEqual(
      "Test from_unixtime with alternative month and two digit year", datesPostEpochDf) {
    frame => frame.select(from_unixtime(col("dates"),"dd/LL/yy HH:mm:ss.SSSSSS"))
  }

  testSparkResultsAreEqual("Test timesub - 4000 seconds", epochDf) {
    frame => frame.selectExpr("cast(dates as timestamp) - (interval 40000 seconds)")
  }

  testSparkResultsAreEqual("Test timesub - 4 day", epochDf) {
    frame => frame.selectExpr("cast(dates as timestamp) - (interval 4 days)")
  }

  testSparkResultsAreEqual("Test timesub - 4 day 1000 seconds", epochDf) {
    frame => frame.selectExpr("cast(dates as timestamp) - (interval 4 days 1000 seconds)")
  }
}