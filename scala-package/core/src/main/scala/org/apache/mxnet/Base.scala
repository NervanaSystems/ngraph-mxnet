/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet

import org.apache.mxnet.util.NativeLibraryLoader
import org.slf4j.{Logger, LoggerFactory}

import scala.Specializable.Group

private[mxnet] object Base {
  private val logger: Logger = LoggerFactory.getLogger("MXNetJVM")

  // type definitions
  class RefInt(val value: Int = 0)
  class RefLong(val value: Long = 0)
  class RefFloat(val value: Float = 0)
  class RefString(val value: String = null)

  type MXUint = Int
  type MXFloat = Float
  type CPtrAddress = Long

  type NDArrayHandle = CPtrAddress
  type FunctionHandle = CPtrAddress
  type DataIterHandle = CPtrAddress
  type DataIterCreator = CPtrAddress
  type KVStoreHandle = CPtrAddress
  type ExecutorHandle = CPtrAddress
  type SymbolHandle = CPtrAddress
  type RecordIOHandle = CPtrAddress
  type RtcHandle = CPtrAddress

  type MXUintRef = RefInt
  type MXFloatRef = RefFloat
  type NDArrayHandleRef = RefLong
  type FunctionHandleRef = RefLong
  type DataIterHandleRef = RefLong
  type DataIterCreatorRef = RefLong
  type KVStoreHandleRef = RefLong
  type ExecutorHandleRef = RefLong
  type SymbolHandleRef = RefLong
  type RecordIOHandleRef = RefLong
  type RtcHandleRef = RefLong

  val MX_REAL_TYPE = DType.Float32

  // The primitives currently supported for NDArray operations
  val MX_PRIMITIVES = new Group ((Double, Float))

  try {
    try {
      tryLoadLibraryOS("mxnet-scala")
    } catch {
      case e: UnsatisfiedLinkError =>
        logger.warn("MXNet Scala native library not found in path. " +
          "Copying native library from the archive. " +
          "Consider installing the library somewhere in the path " +
          "(for Windows: PATH, for Linux: LD_LIBRARY_PATH), " +
          "or specifying by Java cmd option -Djava.library.path=[lib path].")
        logger.warn("LD_LIBRARY_PATH=" + System.getenv("LD_LIBRARY_PATH"))
        logger.warn("java.library.path=" + System.getProperty("java.library.path"))
        NativeLibraryLoader.loadLibrary("mxnet-scala")
    }
  } catch {
    case e: UnsatisfiedLinkError =>
      logger.error("Couldn't find native library mxnet-scala")
      throw e
  }

  val _LIB = new LibInfo
  checkCall(_LIB.nativeLibInit())

  // TODO: shutdown hook won't work on Windows
  Runtime.getRuntime.addShutdownHook(new Thread() {
    override def run(): Unit = {
      notifyShutdown()
    }
  })

  @throws(classOf[UnsatisfiedLinkError])
  private def tryLoadLibraryOS(libname: String): Unit = {
    logger.info(s"Try loading $libname from native path.")
    System.loadLibrary(libname)
  }

  // helper function definitions
  /**
   * Check the return value of C API call
   *
   * This function will raise exception when error occurs.
   * Wrap every API call with this function
   * @param ret return value from API calls
   */
  def checkCall(ret: Int): Unit = {
    if (ret != 0) {
      throw new MXNetError(_LIB.mxGetLastError())
    }
  }

  // Notify MXNet about a shutdown
  private def notifyShutdown(): Unit = {
    checkCall(_LIB.mxNotifyShutdown())
  }

  // Convert ctypes returned doc string information into parameters docstring.
  def ctypes2docstring(
      argNames: Seq[String],
      argTypes: Seq[String],
      argDescs: Seq[String]): String = {

    val params =
      (argNames zip argTypes zip argDescs) map { case ((argName, argType), argDesc) =>
        val desc = if (argDesc.isEmpty) "" else s"\n$argDesc"
        s"$argName : $argType$desc"
      }
    s"Parameters\n----------\n${params.mkString("\n")}\n"
  }
}

class MXNetError(val err: String) extends Exception(err)

// Some type-classes to ease the work in Symbol.random and NDArray.random modules

class SymbolOrScalar[T](val isScalar: Boolean)
object SymbolOrScalar {
  def apply[T](implicit ev: SymbolOrScalar[T]): SymbolOrScalar[T] = ev
  implicit object FloatWitness extends SymbolOrScalar[Float](true)
  implicit object IntWitness extends SymbolOrScalar[Int](true)
  implicit object SymbolWitness extends SymbolOrScalar[Symbol](false)
}

class NDArrayOrScalar[T](val isScalar: Boolean)
object NDArrayOrScalar {
  def apply[T](implicit ev: NDArrayOrScalar[T]): NDArrayOrScalar[T] = ev
  implicit object FloatWitness extends NDArrayOrScalar[Float](true)
  implicit object IntWitness extends NDArrayOrScalar[Int](true)
  implicit object NDArrayWitness extends NDArrayOrScalar[NDArray](false)
}
