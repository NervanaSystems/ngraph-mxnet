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

class ShapeSuite extends FunSuite with BeforeAndAfterAll {
  test("to string") {
    val s = Shape(1, 2, 3)
    assert(s.toString === "(1,2,3)")
  }

  test("equals") {
    assert(Shape(1, 2, 3) === Shape(1, 2, 3))
    assert(Shape(1, 2) != Shape(1, 2, 3))
  }

  test("drop") {
    val s = Shape(1, 2, 3)
    val s2 = s.drop(1)
    assert(s == Shape(1, 2, 3))
    assert(s2 == Shape(2, 3))
    val s3 = s.drop(2)
    assert(s3 == Shape(3))
  }

  test("slice") {
    val s = Shape(1, 2, 3)
    val s2 = s.slice(0, 1)
    assert(s2 == Shape(1))
  }
}
