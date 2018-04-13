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

import org.apache.mxnet.CheckUtils._
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class ModelParallelSuite extends FunSuite with BeforeAndAfterAll {
  test("chain") {
    val n = 2
    val ctx1 = Context.cpu(0)
    val ctx2 = Context.cpu(1)
    val data1 = Symbol.Variable("data1")
    val data2 = Symbol.Variable("data2")
    val data3 = Symbol.Variable("data3")

    var net: Symbol = null
    new AttrScope(Map("ctx_group" -> "dev1")).withScope {
      net = (data1 + data2) * 3
    }

    new AttrScope(Map("ctx_group" -> "dev2")).withScope {
      net = net + data3
    }

    val shape = Shape(4, 5)
    val arr = (0 until n + 1).map(_ => NDArray.empty(shape, ctx1))
    val arrGrad = (0 until n).map(_ => NDArray.empty(shape, ctx1)) :+ NDArray.empty(shape, ctx2)

    val exec1 = net.bind(ctx1,
      args = arr,
      argsGrad = arrGrad,
      gradReq = "write",
      auxStates = Nil,
      group2ctx = Map("dev1" -> ctx1, "dev2" -> ctx2),
      sharedExec = null)

    arr(0).set(1f)
    arr(1).set(2f)
    arr(2).set(3f)

    val arr2 = arr.map(_.copyTo(ctx1))
    val arrGrad2 = arrGrad.map(_.copyTo(ctx1))
    val exec2 = net.bind(ctx1, args = arr2, argsGrad = arrGrad2)

    // Show the execution plan that involves copynode
    // scalastyle:off println
    print(exec1.debugStr)
    // scalastyle:on println

    exec1.forward()
    exec2.forward()
    assert(reldiff(exec1.outputs(0).copyTo(ctx1),
        exec2.outputs(0).copyTo(ctx1)) < 1e-6f)

    val outGrad = NDArray.ones(shape, ctx2)
    exec1.backward(Array(outGrad))
    exec2.backward(Array(outGrad.copyTo(ctx1)))
    (arrGrad zip arrGrad2) foreach { case (a, b) =>
      assert(reldiff(a.copyTo(ctx1), b) < 1e-6f)
    }
  }
}
