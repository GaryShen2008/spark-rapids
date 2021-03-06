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

import ai.rapids.cudf.{ColumnVector, DType, Scalar}

import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.expressions.{Cast, CastBase, Expression, NullIntolerant, TimeZoneAwareExpression}
import org.apache.spark.sql.types._

/** Meta-data for cast and ansi_cast. */
class CastExprMeta[INPUT <: CastBase](
    cast: INPUT,
    ansiEnabled: Boolean,
    conf: RapidsConf,
    parent: Option[RapidsMeta[_, _, _]],
    rule: ConfKeysAndIncompat)
  extends UnaryExprMeta[INPUT](cast, conf, parent, rule) {

  private val castExpr = if (ansiEnabled) "ansi_cast" else "cast"
  private val fromType = cast.child.dataType
  private val toType = cast.dataType

  override def tagExprForGpu(): Unit = {
    if (!GpuCast.canCast(cast.child.dataType, cast.dataType)) {
      willNotWorkOnGpu(s"$castExpr from ${cast.child.dataType} " +
        s"to ${cast.dataType} is not currently supported on the GPU")
    }
    if (!conf.isCastFloatToStringEnabled && toType == DataTypes.StringType &&
      (fromType == DataTypes.FloatType || fromType == DataTypes.DoubleType)) {
      willNotWorkOnGpu("the GPU will use different precision than Java's toString method when " +
        "converting floating point data types to strings and this can produce results that " +
        "differ from the default behavior in Spark.  To enable this operation on the GPU, set" +
        s" ${RapidsConf.ENABLE_CAST_FLOAT_TO_STRING} to true.")
    }
    if (!conf.isCastStringToFloatEnabled && cast.child.dataType == DataTypes.StringType &&
      Seq(DataTypes.FloatType, DataTypes.DoubleType).contains(cast.dataType)) {
      willNotWorkOnGpu("Currently hex values aren't supported on the GPU. Also note " +
        "that casting from string to float types on the GPU returns incorrect results when the " +
        "string represents any number \"1.7976931348623158E308\" <= x < " +
        "\"1.7976931348623159E308\" and \"-1.7976931348623159E308\" < x <= " +
        "\"-1.7976931348623158E308\" in both these cases the GPU returns Double.MaxValue while " +
        "CPU returns \"+Infinity\" and \"-Infinity\" respectively. To enable this operation on " +
        "the GPU, set" + s" ${RapidsConf.ENABLE_CAST_STRING_TO_FLOAT} to true.")
    }
    if (!conf.isCastStringToIntegerEnabled && cast.child.dataType == DataTypes.StringType &&
    Seq(DataTypes.ByteType, DataTypes.ShortType, DataTypes.IntegerType, DataTypes.LongType)
      .contains(cast.dataType)) {
      willNotWorkOnGpu("the GPU will return incorrect results for strings representing" +
        "values greater than Long.MaxValue or less than Long.MinValue.  To enable this " +
        "operation on the GPU, set" +
        s" ${RapidsConf.ENABLE_CAST_STRING_TO_INTEGER} to true.")
    }
  }

  override def convertToGpu(child: GpuExpression): GpuExpression =
    GpuCast(child, toType, ansiEnabled, cast.timeZoneId)
}

object GpuCast {

  /**
   * Regex for identifying strings that contain numeric values that can be casted to integral
   * types. This includes floating point numbers but not numbers containing exponents.
   */
  private val CASTABLE_TO_INT_REGEX = "\\s*[+\\-]?[0-9]*(\\.)?[0-9]+\\s*$"

  /**
   * Regex for identifying strings that contain numeric values that can be casted to integral
   * types when ansi is enabled.
   */
  private val ANSI_CASTABLE_TO_INT_REGEX = "\\s*[+\\-]?[0-9]+\\s*$"

  val INVALID_INPUT_MESSAGE = "Column contains at least one value that is not in the " +
    "required range"

  val INVALID_FLOAT_CAST_MSG = "At least one value is either null or is an invalid number"

  /**
   * Returns true iff we can cast `from` to `to` using the GPU.
   */
  def canCast(from: DataType, to: DataType): Boolean = {
    if (from == to) {
      return true
    }
    from match {
      case BooleanType => to match {
        case ByteType | ShortType | IntegerType | LongType => true
        case FloatType | DoubleType => true
        case TimestampType => true
        case StringType => true
        case _ => false
      }
      case ByteType | ShortType | IntegerType | LongType => to match {
        case BooleanType => true
        case ByteType | ShortType | IntegerType | LongType => true
        case FloatType | DoubleType => true
        case StringType => true
        case TimestampType => true
        case _ => false
      }
      case FloatType | DoubleType => to match {
        case BooleanType => true
        case ByteType | ShortType | IntegerType | LongType => true
        case FloatType | DoubleType => true
        case TimestampType => true
        case StringType => true
        case _ => false
      }
      case DateType => to match {
        case BooleanType => true
        case ByteType | ShortType | IntegerType | LongType => true
        case FloatType | DoubleType => true
        case TimestampType => true
        case StringType => true
        case _ => false
      }
      case TimestampType => to match {
        case BooleanType => true
        case ByteType | ShortType | IntegerType => true
        case LongType => true
        case FloatType | DoubleType => true
        case DateType => true
        case StringType => true
        case _ => false
      }
      case StringType => to match {
        case BooleanType => true
        case ByteType | ShortType | IntegerType | LongType | FloatType | DoubleType => true
        case _ => false
      }
      case _ => false
    }
  }
}

/**
 * Casts using the GPU
 */
case class GpuCast(
    child: GpuExpression,
    dataType: DataType,
    ansiMode: Boolean = false,
    timeZoneId: Option[String] = None)
  extends GpuUnaryExpression with TimeZoneAwareExpression with NullIntolerant {

  override def toString: String = if (ansiMode) {
    s"ansi_cast($child as ${dataType.simpleString})"
  } else {
    s"cast($child as ${dataType.simpleString})"
  }

  override def checkInputDataTypes(): TypeCheckResult = {
    if (Cast.canCast(child.dataType, dataType)) {
      TypeCheckResult.TypeCheckSuccess
    } else {
      TypeCheckResult.TypeCheckFailure(
        s"cannot cast ${child.dataType.catalogString} to ${dataType.catalogString}")
    }
  }

  override def nullable: Boolean = Cast.forceNullable(child.dataType, dataType) || child.nullable

  override def withTimeZone(timeZoneId: String): TimeZoneAwareExpression =
    copy(timeZoneId = Option(timeZoneId))

  /**
   * Under certain conditions during hash partitioning, Spark will attempt to replace casts
   * with semantically equivalent expressions. This method is overridden to prevent Spark
   * from substituting non-GPU expressions.
   */
  override def semanticEquals(other: Expression): Boolean = other match {
    case g: GpuExpression =>
      if (this == g) {
        true
      } else {
        super.semanticEquals(g)
      }
    case _ => false
  }

  // When this cast involves TimeZone, it's only resolved if the timeZoneId is set;
  // Otherwise behave like Expression.resolved.
  override lazy val resolved: Boolean =
    childrenResolved && checkInputDataTypes().isSuccess && (!needsTimeZone || timeZoneId.isDefined)

  private[this] def needsTimeZone: Boolean = Cast.needsTimeZone(child.dataType, dataType)

  override def sql: String = dataType match {
    // HiveQL doesn't allow casting to complex types. For logical plans translated from HiveQL,
    // this type of casting can only be introduced by the analyzer, and can be omitted when
    // converting back to SQL query string.
    case _: ArrayType | _: MapType | _: StructType => child.sql
    case _ => s"CAST(${child.sql} AS ${dataType.sql})"
  }

  override def doColumnar(input: GpuColumnVector): GpuColumnVector = {
    val cudfType = GpuColumnVector.getRapidsType(dataType)

    (input.dataType(), dataType) match {
      case (DateType, BooleanType | _: NumericType) =>
        // casts from date type to numerics are always null
        val scalar = GpuScalar.from(null, dataType)
        try {
          GpuColumnVector.from(scalar, input.getBase.getRowCount.toInt)
        } finally {
          scalar.close()
        }
      case (DateType, StringType) =>
        GpuColumnVector.from(input.getBase.asStrings("%Y-%m-%d"))
      case (TimestampType, FloatType | DoubleType) =>
        val asLongs = input.getBase.castTo(DType.INT64)
        try {
          val microsPerSec = Scalar.fromDouble(1000000)
          try {
            // Use trueDiv to ensure cast to double before division for full precision
            GpuColumnVector.from(asLongs.trueDiv(microsPerSec, cudfType))
          } finally {
            microsPerSec.close()
          }
        } finally {
          asLongs.close()
        }
      case (TimestampType, ByteType | ShortType | IntegerType) =>
        // normally we would just do a floordiv here, but cudf downcasts the operands to
        // the output type before the divide.  https://github.com/rapidsai/cudf/issues/2574
        val asLongs = input.getBase.castTo(DType.INT64)
        try {
          val microsPerSec = Scalar.fromInt(1000000)
          try {
            val cv = asLongs.floorDiv(microsPerSec, DType.INT64)
            try {
              if (ansiMode) {
                dataType match {
                  case IntegerType =>
                    assertValuesInRange(cv, Scalar.fromInt(Int.MinValue),
                      Scalar.fromInt(Int.MaxValue))
                  case ShortType =>
                    assertValuesInRange(cv, Scalar.fromShort(Short.MinValue),
                      Scalar.fromShort(Short.MaxValue))
                  case ByteType =>
                    assertValuesInRange(cv, Scalar.fromByte(Byte.MinValue),
                      Scalar.fromByte(Byte.MaxValue))
                }
              }
              GpuColumnVector.from(cv.castTo(cudfType))
            } finally {
              cv.close()
            }
          } finally {
            microsPerSec.close()
          }
        } finally {
          asLongs.close()
        }
      case (TimestampType, _: LongType) =>
        val asLongs = input.getBase.castTo(DType.INT64)
        try {
          val microsPerSec = Scalar.fromInt(1000000)
          try {
            GpuColumnVector.from(asLongs.floorDiv(microsPerSec, cudfType))
          } finally {
            microsPerSec.close()
          }
        } finally {
          asLongs.close()
        }
      case (TimestampType, StringType) =>
        GpuColumnVector.from(input.getBase.asStrings("%Y-%m-%d %H:%M:%S.%3f"))

      // ansi cast from larger-than-integer integral types, to integer
      case (LongType, IntegerType) if ansiMode =>
        assertValuesInRange(input.getBase, Scalar.fromInt(Int.MinValue),
          Scalar.fromInt(Int.MaxValue))
        GpuColumnVector.from(input.getBase.castTo(cudfType))

      // ansi cast from larger-than-short integral types, to short
      case (LongType|IntegerType, ShortType) if ansiMode =>
        assertValuesInRange(input.getBase, Scalar.fromShort(Short.MinValue),
          Scalar.fromShort(Short.MaxValue))
        GpuColumnVector.from(input.getBase.castTo(cudfType))

      // ansi cast from larger-than-byte integral types, to byte
      case (LongType|IntegerType|ShortType, ByteType) if ansiMode =>
        assertValuesInRange(input.getBase, Scalar.fromByte(Byte.MinValue),
          Scalar.fromByte(Byte.MaxValue))
        GpuColumnVector.from(input.getBase.castTo(cudfType))

      // ansi cast from floating-point types, to byte
      case (FloatType|DoubleType, ByteType) if ansiMode =>
        assertValuesInRange(input.getBase, Scalar.fromByte(Byte.MinValue),
          Scalar.fromByte(Byte.MaxValue))
        GpuColumnVector.from(input.getBase.castTo(cudfType))

      // ansi cast from floating-point types, to short
      case (FloatType|DoubleType, ShortType) if ansiMode =>
        assertValuesInRange(input.getBase, Scalar.fromShort(Short.MinValue),
          Scalar.fromShort(Short.MaxValue))
        GpuColumnVector.from(input.getBase.castTo(cudfType))

      // ansi cast from floating-point types, to integer
      case (FloatType|DoubleType, IntegerType) if ansiMode =>
        assertValuesInRange(input.getBase, Scalar.fromInt(Int.MinValue),
          Scalar.fromInt(Int.MaxValue))
        GpuColumnVector.from(input.getBase.castTo(cudfType))

      // ansi cast from floating-point types, to long
      case (FloatType|DoubleType, LongType) if ansiMode =>
        assertValuesInRange(input.getBase, Scalar.fromLong(Long.MinValue),
          Scalar.fromLong(Long.MaxValue))
        GpuColumnVector.from(input.getBase.castTo(cudfType))

      case (FloatType | DoubleType, TimestampType) =>
        // Spark casting to timestamp from double assumes value is in microseconds
        withResource(Scalar.fromInt(1000000)) { microsPerSec =>
          withResource(input.getBase.nansToNulls()) { inputWithNansToNull =>
            withResource(FloatUtils.infinityToNulls(inputWithNansToNull)) {
              inputWithoutNanAndInfinity =>
                withResource(inputWithoutNanAndInfinity.mul(microsPerSec)) { inputTimesMicrosCv =>
                  GpuColumnVector.from(inputTimesMicrosCv.castTo(DType.TIMESTAMP_MICROSECONDS))
                }
            }
          }
        }
      case (_: NumericType, TimestampType) =>
        // Spark casting to timestamp assumes value is in seconds, but timestamps
        // are tracked in microseconds.
        val timestampSecs = input.getBase.castTo(DType.TIMESTAMP_SECONDS)
        try {
          GpuColumnVector.from(timestampSecs.castTo(cudfType))
        } finally {
          timestampSecs.close();
        }
        // Float.NaN => Int is casted to a zero but float.NaN => Long returns a small negative
        // number Double.NaN => Int | Long, returns a small negative number so Nans have to be
        // converted to zero first
      case (FloatType, LongType) | (DoubleType, IntegerType | LongType) =>
        withResource(FloatUtils.nanToZero(input.getBase)) { inputWithNansToZero =>
          GpuColumnVector.from(inputWithNansToZero.castTo(cudfType))
        }
      case (FloatType|DoubleType, StringType) =>
        castFloatingTypeToString(input)
      case (StringType, BooleanType | ByteType | ShortType | IntegerType | LongType | FloatType
                        | DoubleType) =>
        withResource(input.getBase.strip()) { trimmed =>
          dataType match {
            case BooleanType =>
              castStringToBool(trimmed, ansiMode)
            case FloatType | DoubleType =>
              castStringToFloats(trimmed, ansiMode, cudfType)
            case ByteType | ShortType | IntegerType | LongType =>
              // filter out values that are not valid longs or nulls
              val regex = if (ansiMode) {
                GpuCast.ANSI_CASTABLE_TO_INT_REGEX
              } else {
                GpuCast.CASTABLE_TO_INT_REGEX
              }
              val longStrings = withResource(trimmed.matchesRe(regex)) { regexMatches =>
                if (ansiMode) {
                  withResource(regexMatches.all()) { allRegexMatches =>
                    if (!allRegexMatches.getBoolean) {
                      throw new NumberFormatException(GpuCast.INVALID_INPUT_MESSAGE)
                    }
                  }
                }
                withResource(Scalar.fromNull(DType.STRING)) { nullString =>
                  regexMatches.ifElse(trimmed, nullString)
                }
              }
              // cast to specific integral type after filtering out values that are not in range
              // for that type. Note that the scalar values here are named parameters so are not
              // created until they are needed
              withResource(longStrings) { longStrings =>
                cudfType match {
                  case DType.INT8 =>
                    castStringToIntegralType(longStrings, DType.INT8,
                      Scalar.fromInt(Byte.MinValue), Scalar.fromInt(Byte.MaxValue))
                  case DType.INT16 =>
                    castStringToIntegralType(longStrings, DType.INT16,
                      Scalar.fromInt(Short.MinValue), Scalar.fromInt(Short.MaxValue))
                  case DType.INT32 =>
                    castStringToIntegralType(longStrings, DType.INT32,
                      Scalar.fromInt(Int.MinValue), Scalar.fromInt(Int.MaxValue))
                  case DType.INT64 =>
                    GpuColumnVector.from(longStrings.castTo(DType.INT64))
                  case _ =>
                    throw new IllegalStateException("Invalid integral type")
                }
              }
          }
        }

      case _ =>
        GpuColumnVector.from(input.getBase.castTo(cudfType))
    }
  }

  /**
   * Asserts that all values in a column are within the specfied range.
   *
   * @param values ColumnVector
   * @param minValue Named parameter for function to create Scalar representing range minimum value
   * @param maxValue Named parameter for function to create Scalar representing range maximum value
   * @throws IllegalStateException if any values in the column are not within the specified range
   */
  private def assertValuesInRange(values: ColumnVector,
    minValue: => Scalar,
    maxValue: => Scalar): Unit = {

    def throwIfAny(cv: ColumnVector): Unit = {
      withResource(cv) { cv =>
        withResource(cv.any()) { isAny =>
          if (isAny.getBoolean) {
            throw new IllegalStateException(GpuCast.INVALID_INPUT_MESSAGE)
          }
        }
      }
    }

    withResource(minValue) { minValue =>
      throwIfAny(values.lessThan(minValue))
    }

    withResource(maxValue) { maxValue =>
      throwIfAny(values.greaterThan(maxValue))
    }
  }

  private def castFloatingTypeToString(input: GpuColumnVector): GpuColumnVector = {
    withResource(input.getBase.castTo(DType.STRING)) { cudfCast =>

      // replace "e+" with "E"
      val replaceExponent = withResource(Scalar.fromString("e+")) { cudfExponent =>
        withResource(Scalar.fromString("E")) { sparkExponent =>
          cudfCast.stringReplace(cudfExponent, sparkExponent)
        }
      }

      // replace "Inf" with "Infinity"
      withResource(replaceExponent) { replaceExponent =>
        withResource(Scalar.fromString("Inf")) { cudfInf =>
          withResource(Scalar.fromString("Infinity")) { sparkInfinity =>
            GpuColumnVector.from(replaceExponent.stringReplace(cudfInf, sparkInfinity))
          }
        }
      }
    }
  }

  private def castStringToBool(input: ColumnVector, ansiEnabled: Boolean): GpuColumnVector = {
    val trueStrings = Seq("t", "true", "y", "yes", "1")
    val falseStrings = Seq("f", "false", "n", "no", "0")
    val boolStrings = trueStrings ++ falseStrings

    // determine which values are valid bool strings
    withResource(ColumnVector.fromStrings(boolStrings: _*)) { boolStrings =>
      withResource(input.contains(boolStrings)) { validBools =>
        // in ansi mode, fail if any values are not valid bool strings
        if (ansiEnabled) {
          withResource(validBools.all()) { isAllBool =>
            if (!isAllBool.getBoolean) {
              throw new IllegalStateException(GpuCast.INVALID_INPUT_MESSAGE)
            }
          }
        }
        // replace non-boolean values with null
        withResource(Scalar.fromNull(DType.STRING)) { nullString =>
          withResource(validBools.ifElse(input, nullString)) { sanitizedInput =>
            // return true, false, or null, as appropriate
            withResource(ColumnVector.fromStrings(trueStrings: _*)) { cvTrue =>
              GpuColumnVector.from(sanitizedInput.contains(cvTrue))
            }
          }
        }
      }
    }
  }

  def castStringToFloats(
      input: ColumnVector,
      ansiEnabled: Boolean, dType: DType): GpuColumnVector = {

    // TODO: since cudf doesn't support case-insensitive regex, we have to generate all
    //  possible strings. But these should cover most of the cases
    val POS_INF_REGEX = "^[+]?(?:infinity|inf|Infinity|Inf|INF|INFINITY)$"
    val NEG_INF_REGEX = "^[\\-](?:infinity|inf|Infinity|Inf|INF|INFINITY)$"
    val NAN_REGEX = "^(?:nan|NaN|NAN)$"


//      1. convert the different infinities to "Inf"/"-Inf" which is the only variation cudf
//         understands
//      2. identify the nans
//      3. identify the floats. "nan", "null" and letters are not considered floats
//      4. if ansi is enabled we want to throw and exception if the string is neither float nor nan
//      5. convert everything thats not floats to null
//      6. set the indices where we originally had nans to Float.NaN
//
//      NOTE Limitation: "1.7976931348623159E308" and "-1.7976931348623159E308" are not considered
//      Inf even though spark does

    if (ansiEnabled && input.hasNulls()) {
      throw new NumberFormatException(GpuCast.INVALID_FLOAT_CAST_MSG)
    }
    // First replace different spellings/cases of infinity with Inf and -Infinity with -Inf
    val posInfReplaced = withResource(input.matchesRe(POS_INF_REGEX)) { containsInf =>
      withResource(Scalar.fromString("Inf")) { inf =>
        containsInf.ifElse(inf, input)
      }
    }
    val withPosNegInfinityReplaced = withResource(posInfReplaced) { withPositiveInfinityReplaced =>
      withResource(withPositiveInfinityReplaced.matchesRe(NEG_INF_REGEX)) { containsNegInf =>
        withResource(Scalar.fromString("-Inf")) { negInf =>
          containsNegInf.ifElse(negInf, withPositiveInfinityReplaced)
        }
      }
    }
    //Now identify the different variations of nans
    withResource(withPosNegInfinityReplaced.matchesRe(NAN_REGEX)) { isNan =>
      // now check if the values are floats
      withResource(withPosNegInfinityReplaced.isFloat()) { isFloat =>
        if (ansiEnabled) {
          withResource(isNan.not()) { notNan =>
            withResource(isFloat.not()) { notFloat =>
              withResource(notFloat.and(notNan)) { notFloatAndNotNan =>
                withResource(notFloatAndNotNan.any()) { notNanAndNotFloat =>
                  if (notNanAndNotFloat.getBoolean()) {
                    throw new NumberFormatException(GpuCast.INVALID_FLOAT_CAST_MSG)
                  }
                }
              }
            }
          }
        }
        withResource(withPosNegInfinityReplaced.castTo(dType)) { casted =>
          withResource(Scalar.fromNull(dType)) { nulls =>
            withResource(isFloat.ifElse(casted, nulls)) { floatsOnly =>
              withResource(FloatUtils.getNanScalar(dType)) { nan =>
                GpuColumnVector.from(isNan.ifElse(nan, floatsOnly))
              }
            }
          }
        }
      }
    }
  }

  /**
   * Cast column of long values to a smaller integral type (bytes, short, int).
   *
   * @param longStrings Long values in string format
   * @param castToType Type to cast to
   * @param minValue Named parameter for function to create Scalar representing range minimum value
   * @param maxValue Named parameter for function to create Scalar representing range maximum value
   * @return Values cast to specified integral type
   */
  private def castStringToIntegralType(longStrings: ColumnVector,
      castToType: DType,
      minValue: => Scalar,
      maxValue: => Scalar): GpuColumnVector = {

    // evaluate min and max named parameters once since they are used in multiple places
    withResource(minValue) { minValue: Scalar =>
      withResource(maxValue) { maxValue: Scalar =>
        withResource(Scalar.fromNull(DType.INT64)) { nulls =>
          withResource(longStrings.castTo(DType.INT64)) { values =>

            // replace values less than minValue with null
            val gtEqMinOrNull = withResource(values.greaterOrEqualTo(minValue)) { isGtEqMin =>
              if (ansiMode) {
                withResource(isGtEqMin.all()) { all =>
                  if (!all.getBoolean) {
                    throw new NumberFormatException(GpuCast.INVALID_INPUT_MESSAGE)
                  }
                }
              }
              isGtEqMin.ifElse(values, nulls)
            }

            // replace values greater than maxValue with null
            val ltEqMaxOrNull = withResource(gtEqMinOrNull) { gtEqMinOrNull =>
              withResource(gtEqMinOrNull.lessOrEqualTo(maxValue)) { isLtEqMax =>
                if (ansiMode) {
                  withResource(isLtEqMax.all()) { all =>
                    if (!all.getBoolean) {
                      throw new NumberFormatException(GpuCast.INVALID_INPUT_MESSAGE)
                    }
                  }
                }
                isLtEqMax.ifElse(gtEqMinOrNull, nulls)
              }
            }

            // cast the final values
            withResource(ltEqMaxOrNull) { ltEqMaxOrNull =>
              GpuColumnVector.from(ltEqMaxOrNull.castTo(castToType))
            }
          }
        }
      }

    }
  }

}
