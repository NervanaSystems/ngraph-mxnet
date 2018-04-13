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

class RandomSuite extends FunSuite with BeforeAndAfterAll {
  test("uniform on cpu") {
    Context.cpu().withScope {
      val (a, b) = (-10, 10)
      val shape = Shape(100, 100)
      Random.seed(128)
      val un1 = Random.uniform(a, b, shape)
      Random.seed(128)
      val un2 = Random.uniform(a, b, shape)
      assert(un1 === un2)
      assert(Math.abs(un1.toArray.sum / un1.size - (a + b) / 2f) < 0.1)
    }
  }

  test("normal on cpu") {
    val (mu, sigma) = (10f, 2f)
    val shape = Shape(100, 100)
    Random.seed(128)
    val ret1 = Random.normal(mu, sigma, shape)
    Random.seed(128)
    val ret2 = Random.normal(mu, sigma, shape)
    assert(ret1 === ret2)

    val array = ret1.toArray
    val mean = array.sum / ret1.size
    val devs = array.map(score => (score - mean) * (score - mean))
    val stddev = Math.sqrt(devs.sum / ret1.size)

    assert(Math.abs(mean - mu) < 0.1)
    assert(Math.abs(stddev - sigma) < 0.1)
  }
}
