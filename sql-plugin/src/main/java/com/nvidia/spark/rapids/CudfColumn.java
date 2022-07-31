/*
 Copyright (c) 2021 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package com.nvidia.spark.rapids;

import java.util.Base64;

import com.nvidia.spark.rapids.jni.CudaIPC;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;

/**
 * This class is composing of base data with Apache Arrow format from Cudf ColumnVector.
 * It will be used to generate the cuda array interface.
 */
public class CudfColumn {

  private final long dataPtr; //  gpu data buffer address
  private final long shape;   // row count
  private final long validPtr; // gpu valid buffer address
  private final int typeSize; // type size in bytes
  private final String typeStr; // follow array interface spec
  private final long nullCount; // null count

  private String arrayInterface = null; // the cuda array interface

  public static CudfColumn from(ColumnVector cv) {
    System.out.println("CudfColumn: " + cv);
    BaseDeviceMemoryBuffer dataBuffer = cv.getDeviceBufferFor(BufferType.DATA);
    BaseDeviceMemoryBuffer validBuffer = cv.getDeviceBufferFor(BufferType.VALIDITY);
    long validPtr = 0;
    if (validBuffer != null) {
      validPtr = validBuffer.getAddress();
    }
    DType dType = cv.getType();
    String typeStr = "";
    if (dType == DType.FLOAT32 || dType == DType.FLOAT64 ||
        dType == DType.TIMESTAMP_DAYS || dType == DType.TIMESTAMP_MICROSECONDS ||
        dType == DType.TIMESTAMP_MILLISECONDS || dType == DType.TIMESTAMP_NANOSECONDS ||
        dType == DType.TIMESTAMP_SECONDS) {
      typeStr = "<f" + dType.getSizeInBytes();
    } else if (dType == DType.BOOL8 || dType == DType.INT8 || dType == DType.INT16 ||
        dType == DType.INT32 || dType == DType.INT64) {
      typeStr = "<i" + dType.getSizeInBytes();
    } else {
      // Unsupported type.
      throw new IllegalArgumentException("Unsupported data type: " + dType);
    }

    return new CudfColumn(dataBuffer.getAddress(), cv.getRowCount(), validPtr,
      dType.getSizeInBytes(), typeStr, cv.getNullCount());
  }

  private CudfColumn(long dataPtr, long shape, long validPtr, int typeSize, String typeStr,
                     long nullCount) {
    this.dataPtr = dataPtr;
    this.shape = shape;
    this.validPtr = validPtr;
    this.typeSize = typeSize;
    this.typeStr = typeStr;
    this.nullCount = nullCount;
  }

  public String getArrayInterfaceJson() {
    // There is no race-condition
    if (arrayInterface == null) {
      arrayInterface = buildColumnCudaIpcInfo(this);
    }
    return arrayInterface;
  }

  public long getDataPtr() {
    return dataPtr;
  }

  public long getShape() {
    return shape;
  }

  public long getValidPtr() {
    return validPtr;
  }

  public int getTypeSize() {
    return typeSize;
  }

  public String getTypeStr() {
    return typeStr;
  }

  public long getNullCount() {
    return nullCount;
  }


  /** build the whole column information including data and valid info */
  private String buildColumnCudaIpcInfo(CudfColumn column) {
    if (column.getDataPtr() == 0) {
      throw new IllegalArgumentException("Empty column data is NOT accepted!");
    }
    if (column.getTypeStr() == null || column.getTypeStr().isEmpty()) {
      throw new IllegalArgumentException("Empty type string is NOT accepted!");
    }

    StringBuilder builder = new StringBuilder();
    String colData = buildMetaObject(column.getDataPtr(), column.getShape(),
        column.getTypeStr());
    builder.append("{");
    builder.append(colData);
    if (column.getValidPtr() != 0 && column.getNullCount() != 0) {
      String validString = buildMetaObject(column.getValidPtr(), column.getShape(), "<t1");
      builder.append(",\"mask\":");
      builder.append("{");
      builder.append(validString);
      builder.append("}");
    }
    builder.append("}");
    return builder.toString();
  }

  /** build the base information of a column */
  private String buildMetaObject(long ptr, long shape, final String typeStr) {

    byte[] bytes = CudaIPC.getCudaIpcMemHandle(ptr);
    String cudaIpcInfo = Base64.getEncoder().encodeToString(bytes);

    StringBuilder builder = new StringBuilder();
    builder.append("\"shape\":[" + shape + "],");
    builder.append("\"data\":[\"" + cudaIpcInfo + "\"," + "false" + "],");
    builder.append("\"typestr\":\"" + typeStr + "\",");
    builder.append("\"version\":" + 1);
    return builder.toString();
  }

}
