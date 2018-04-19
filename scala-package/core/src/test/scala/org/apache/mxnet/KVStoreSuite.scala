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

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class KVStoreSuite extends FunSuite with BeforeAndAfterAll {
  test("init and pull") {
    val kv = KVStore.create()
    val shape = Shape(2, 1)
    val ndArray = NDArray.zeros(shape)

    kv.init("3", NDArray.ones(shape))
    kv.pull("3", ndArray)
    assert(ndArray.toArray === Array(1f, 1f))
  }

  test("push and pull") {
    val kv = KVStore.create()
    val shape = Shape(2, 1)
    val ndArray = NDArray.zeros(shape)

    kv.init("3", NDArray.ones(shape))
    kv.push("3", NDArray.ones(shape) * 4)
    kv.pull("3", ndArray)
    assert(ndArray.toArray === Array(4f, 4f))
  }

  test("test aggregate") {
    val shape = Shape(4, 4)
    val keys = Array("b", "c", "d")
    val kv = KVStore.create()
    kv.init("a", NDArray.zeros(shape))
    kv.init(keys, Array.fill(keys.length)(NDArray.zeros(shape)))
    val numDevs = 4
    val devs = (0 until numDevs).map(Context.cpu(_))
    val vals = devs.map(d => NDArray.ones(shape, d)).toArray
    kv.push("a", vals)
    kv.pull("a", outs = vals)
    assert(vals.map(v => v.toArray.map(x => x - numDevs).sum).sum == 0f)

    val valss = keys.map { k =>
      val tmpVals = devs.map(d => NDArray.ones(shape, d) * 2f).toArray
      kv.push(k, tmpVals)
      kv.pull(k, outs = tmpVals)
      tmpVals
    }.flatten
    assert(valss.map(v => v.toArray.map(x => x - numDevs * 2f).sum).sum == 0f)
  }

  test("updater runs when push") {
    val kv = KVStore.create()
    val updater = new MXKVStoreUpdater {
      override def update(key: Int, input: NDArray, stored: NDArray): Unit = {
        // scalastyle:off println
        println(s"update on key $key")
        // scalastyle:on println
        stored += input * 2
      }
      override def dispose(): Unit = {}
    }
    kv.setUpdater(updater)

    val shape = Shape(2, 1)
    val ndArray = NDArray.zeros(shape)

    kv.init("3", NDArray.ones(shape) * 4)
    kv.pull("3", ndArray)
    assert(ndArray.toArray === Array(4f, 4f))

    kv.push("3", NDArray.ones(shape))
    kv.pull("3", ndArray)
    assert(ndArray.toArray === Array(6f, 6f))
  }

  test("get type") {
    val kv = KVStore.create("local")
    assert(kv.`type` === "local")
  }

  test("get numWorkers and rank") {
    val kv = KVStore.create("local")
    assert(kv.numWorkers === 1)
    assert(kv.rank === 0)
  }
}
