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

class AttrScopeSuite extends FunSuite with BeforeAndAfterAll {
  test("attr basic") {
    val (data, gdata) =
    AttrScope(Map("group" -> "4", "data" -> "great")).withScope {
      val data = Symbol.Variable("data", attr = Map("dtype" -> "data", "group" -> "1"))
      val gdata = Symbol.Variable("data2")
      (data, gdata)
    }
    assert(gdata.attr("group").get === "4")
    assert(data.attr("group").get === "1")

    val exceedScopeData = Symbol.Variable("data3")
    assert(exceedScopeData.attr("group") === None, "No group attr in global attr scope")
  }
}
