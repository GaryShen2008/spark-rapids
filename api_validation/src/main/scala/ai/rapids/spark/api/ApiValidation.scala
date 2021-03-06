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

package ai.rapids.spark.api
import ai.rapids.spark._

import scala.reflect.runtime.{universe => ru}
import ru.MethodSymbolTag
import scala.reflect.api

object ApiValidation {
  // Method to take in a fully qualified case class name and return case class accessors in the
  // form of list of tuples(parameterName, Type)
  def getCaseClassAccessors[T: ru.TypeTag]: List[ru.MethodSymbol] = ru.typeOf[T].members.sorted
    .collect {
    case m: ru.MethodSymbol if m.isCaseAccessor => m
  }

  // Method to convert string to TypeTag where string is fully qualified class Name
  def stringToTypeTag[A](execName: String): ru.TypeTag[A] = {
    val execNameObject = Class.forName(execName)  // obtain object from execName
    val runTimeMirror = ru.runtimeMirror(execNameObject.getClassLoader)  // obtain runtime mirror
    val classSym = runTimeMirror.staticClass(execName)  // obtain class symbol for `execNameObject`
    val tpe = classSym.selfType  // obtain type object for `execNameObject`
    // create a type tag which contains above type object
    ru.TypeTag(runTimeMirror, new api.TypeCreator {
      def apply[U <: api.Universe with Singleton](m: api.Mirror[U]): U#Type =
        if (m eq runTimeMirror) tpe.asInstanceOf[U#Type]
        else throw new IllegalArgumentException(s"Type tag defined in $runTimeMirror cannot be " +
          s"migrated to other mirrors.")
    })
  }

  def printHeaders(a: String): Unit = {
    println("\n*******************************************************************************" +
      "*****************")
    println(a)
    println("Spark parameters                                                    Plugin parameters")
    println("----------------------------------------------------------------------------------" +
      "--------------")
  }

  def main(args: Array[String]): Unit ={
    val gpuExecs = GpuOverrides.execs // get all the execs
    // get the keys Eg: class org.apache.spark.sql.execution.aggregate.HashAggregateExec
    val gpuKeys = gpuExecs.keys
    var printNewline = false

    gpuKeys.map(x => x.getName).foreach { e =>
      // Get SparkExecs argNames and types
      val sparkTypes = stringToTypeTag(e)
      val sparkParameters = getCaseClassAccessors(sparkTypes).map(m => m.name -> m.info)

      // Get GpuExecs argNames and Types.
      // Note that for some there is no 1-1 mapping between names
      // Some Execs are in different packages.
      val execType = sparkTypes.tpe.toString.split('.').last

      val gpu = execType match {
        case "BroadcastHashJoinExec" | "BroadcastExchangeExec" =>
          s"org.apache.spark.sql.execution.Gpu" + execType
        case "FileSourceScanExec" => s"org.apache.spark.sql.rapids.Gpu" + execType
        case "SortMergeJoinExec" => s"ai.rapids.spark.GpuShuffledHashJoinExec"
        case "SortAggregateExec" => s"ai.rapids.spark.GpuHashAggregateExec"
        case _ => s"ai.rapids.spark.Gpu" + execType
      }

      // TODO: Add error handling if Type is not present
      val gpuTypes=stringToTypeTag(gpu)

      val sparkToGpuExecMap = Map(
        "org.apache.spark.sql.catalyst.expressions.Expression" ->
          "ai.rapids.spark.GpuExpression",
        "org.apache.spark.sql.catalyst.expressions.NamedExpression" ->
          "ai.rapids.spark.GpuExpression",
        "org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression" ->
            "org.apache.spark.sql.rapids.GpuAggregateExpression",
        "org.apache.spark.sql.catalyst.expressions.AttributeReference" ->
          "ai.rapids.spark.GpuAttributeReference",
        "org.apache.spark.sql.execution.command.DataWritingCommand" ->
          "ai.rapids.spark.GpuDataWritingCommand",
        "org.apache.spark.sql.execution.joins.BuildSide" ->
          "org.apache.spark.sql.execution.joins.BuildSide")
          .withDefaultValue("sparkKeyNotPresent")

      val gpuParameters = getCaseClassAccessors(gpuTypes).map(m => m.name -> m.info)

      // TODO: Redirect the output to a log file
      if (sparkParameters.length != gpuParameters.length) {
        if (!printNewline) {
          println("\n")
          printNewline = true
        }
        printHeaders(s"Parameter lengths don't match between Execs\nSparkExec - " +
          sparkTypes.toString().replace("TypeTag", "") + "\nGpuExec - " +
          gpuTypes.toString().replace("TypeTag", "") + "\nSpark code has " +
          sparkParameters.length + " parameters where as plugin code has " + gpuParameters.length +
          " parameters")
        val paramLength = if (sparkParameters.length > gpuParameters.length) {
          sparkParameters.length
        } else {
          gpuParameters.length
        }
        // Print sparkExec parameter type and gpuExec parameter type in same line
        for (i <- 0 until paramLength) {
          if (i < sparkParameters.length) {
            print(s"%-65s".format(sparkParameters(i)._2.toString.substring(3)) + " | ")
          } else {
            print(s"%-65s".format(" ")+ " | ")
          }
          if (i < gpuParameters.length) {
            print(s"%s".format(gpuParameters(i)._2.toString.substring(3)))
          }
          println
        }
      } else {
        // Here we need to extract the types from each GPUExec and Spark Exec and
        // compare. For Eg: GpuExpression with Expression.
        var isFirst = true
        for (i <- sparkParameters.indices) {
          val sparkSimpleName = sparkParameters(i)._2.toString.substring(3)
          val gpuSimpleName = gpuParameters(i)._2.toString.substring(3)

          val sparkKey = sparkSimpleName.split('[').last.replace("]", "")
          val gpuKey = sparkSimpleName.replace(sparkKey, sparkToGpuExecMap(sparkKey))

          if (sparkParameters(i)._2 != gpuParameters(i)._2 && gpuKey != gpuSimpleName) {
            if (isFirst) {
              printHeaders(s"Types differ for below parameters in this Exec\nSparkExec  - " +
                sparkTypes.toString().replace("TypeTag", "") + "\nGpuExec - " +
                gpuTypes.toString().replace("TypeTag", ""))
              isFirst = false
            }
            print(s"%-65s".format(sparkSimpleName) + " | ")
            println(s"%s".format(gpuSimpleName)) + "\n"
          }
        }
      }
    }
  }
}
