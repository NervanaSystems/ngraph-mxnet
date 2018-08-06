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

class TestUtil {

  /**
    * Allow override of data path. Default is <current directory>/data
    * @return Data direcotry path ()may be relative)
    */
  def getDataDirectory: String = {
    var dataDir = System.getenv("MXNET_HOME")
    if(dataDir == null) {
      dataDir = "data"
    } else {
      if (dataDir.isEmpty) {
        dataDir = "data"
      }
    }
    dataDir
  }

  /**
    * Create data file path based upon getDataDirectory
    * @param relFile
    * @return file path
    */
  def dataFile(relFile: String): String = {
    getDataDirectory + "/" + relFile
  }

}
