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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, NamedExpression, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.window.WindowExec
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.rapids.GpuAggregateExpression
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.vectorized.ColumnarBatch

class GpuWindowExecMeta(windowExec: WindowExec,
                        conf: RapidsConf,
                        parent: Option[RapidsMeta[_, _, _]],
                        rule: ConfKeysAndIncompat)
  extends SparkPlanMeta[WindowExec](windowExec, conf, parent, rule) {

  val windowExpressions: Seq[ExprMeta[NamedExpression]] =
    windowExec.windowExpression.map(GpuOverrides.wrapExpr(_, conf, Some(this)))

  override def tagPlanForGpu(): Unit = {

    // Implementation depends on receiving a `NamedExpression` wrapped WindowExpression.
    windowExpressions.map(meta => meta.wrapped)
      .filter(expr => !expr.isInstanceOf[NamedExpression])
      .foreach(_ => willNotWorkOnGpu(because = "Unexpected query plan with Windowing functions; " +
        "cannot convert for GPU execution. " +
        "(Detail: WindowExpression not wrapped in `NamedExpression`.)"))

  }

  override def convertToGpu(): GpuExec = {
    GpuWindowExec(
      windowExpressions.map(_.convertToGpu()),
      childPlans.head.convertIfNeeded()
    )
  }
}

case class GpuWindowExec(windowExpressionAliases: Seq[GpuExpression],
                         child: SparkPlan
                        ) extends UnaryExecNode with GpuExec {

  override def output: Seq[Attribute] =
    child.output ++ windowExpressionAliases.map(_.asInstanceOf[NamedExpression].toAttribute)

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override protected def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException(s"Row-based execution should not happen, in $this.")

  private def bindReferences() : Seq[GpuExpression] = {

    // Address bindings for all expressions evaluated by WindowExec.
    val boundProjectList = windowExpressionAliases.map(
      alias => GpuBindReferences.bindReference(alias, child.output)
    )

    // Bind aggregation column.
    boundProjectList.map(
      expr => expr.transform {
        case windowExpr: GpuWindowExpression =>
          val boundAggExpression = GpuBindReferences.bindReference(
            windowExpr.windowFunction match {
              case aggExpression: GpuAggregateExpression =>
                aggExpression.aggregateFunction.inputProjection.head
              case _ : GpuRowNumber => GpuLiteral(1, IntegerType)
              case anythingElse =>
                throw new IllegalStateException(s"Unexpected window operation " +
                  s"${anythingElse.prettyName}")
            },
            child.output)
          windowExpr.setBoundAggCol(boundAggExpression)
          windowExpr
      }.asInstanceOf[GpuExpression]
    )
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {

    val boundOutputProjectList = bindReferences()

    val input = child.executeColumnar()
    input.map {
      cb => {

        var originalCols: Array[GpuColumnVector] = null
        var aggCols     : Array[GpuColumnVector] = null

        try {
          originalCols = GpuColumnVector.extractColumns(cb)
          originalCols.foreach(_.incRefCount())
          aggCols = boundOutputProjectList.map(
            _.columnarEval(cb).asInstanceOf[GpuColumnVector]).toArray

          new ColumnarBatch(originalCols ++ aggCols, cb.numRows())
        } finally {
          cb.close()
        }
      }
    }
  }
}
