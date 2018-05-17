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

package org.apache.mxnet.train

import org.apache.mxnet.optimizer.SGD
import org.apache.mxnet._
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer
import scala.sys.process._

class ConvSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[ConvSuite])

  private var tu = new TestUtil

  test("train mnist") {
    // symbol net
    val batchSize = 100

    val data = Symbol.Variable("data")
    val conv1 = Symbol.Convolution(name = "conv1")()(Map("data" -> data, "num_filter" -> 32,
                                                         "kernel" -> (3, 3), "stride" -> (2, 2)))
    val bn1 = Symbol.BatchNorm(name = "bn1")()(Map("data" -> conv1))
    val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> bn1, "act_type" -> "relu"))
    val mp1 = Symbol.Pooling(name = "mp1")()(Map("data" -> act1, "kernel" -> (2, 2),
                                                 "stride" -> (2, 2), "pool_type" -> "max"))

    val conv2 = Symbol.Convolution(name = "conv2")()(Map("data" -> mp1, "num_filter" -> 32,
                                                         "kernel" -> (3, 3), "stride" -> (2, 2)))
    val bn2 = Symbol.BatchNorm(name = "bn2")()(Map("data" -> conv2))
    val act2 = Symbol.Activation(name = "relu2")()(Map("data" -> bn2, "act_type" -> "relu"))
    val mp2 = Symbol.Pooling(name = "mp2")()(Map("data" -> act2, "kernel" -> (2, 2),
                                                 "stride" -> (2, 2), "pool_type" -> "max"))

    val fl = Symbol.Flatten(name = "flatten")()(Map("data" -> mp2))
    val fc2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> fl, "num_hidden" -> 10))
    val softmax = Symbol.SoftmaxOutput(name = "sm")()(Map("data" -> fc2))

    // get data
    "./scripts/get_mnist_data.sh" !
    val trainDataIter = IO.MNISTIter(Map(
      "image" -> tu.dataFile("train-images-idx3-ubyte"),
      "label" -> tu.dataFile("train-labels-idx1-ubyte"),
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0",
      "seed" -> "10"))

    val valDataIter = IO.MNISTIter(Map(
      "image" -> tu.dataFile("t10k-images-idx3-ubyte"),
      "label" -> tu.dataFile("t10k-labels-idx1-ubyte"),
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0", "silent" -> "0"))

    val model = FeedForward.newBuilder(softmax)
          .setContext(Context.cpu())
          .setNumEpoch(1)
          .setOptimizer(new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
          .setTrainData(trainDataIter)
          .setEvalData(valDataIter)
          .build()
    logger.info("Finish fit ...")

    val probArrays = model.predict(valDataIter)
    assert(probArrays.length === 1)
    val prob = probArrays(0)
    logger.info("Finish predict ...")

    valDataIter.reset()
    val labels = ListBuffer.empty[NDArray]
    while (valDataIter.hasNext) {
      val evalData = valDataIter.next()
      labels += evalData.label(0).copy()
    }
    val y = NDArray.concatenate(labels)

    val py = NDArray.argmax_channel(prob)
    assert(y.shape === py.shape)

    var numCorrect = 0
    var numInst = 0
    for ((labelElem, predElem) <- y.toArray zip py.toArray) {
      if (labelElem == predElem) {
        numCorrect += 1
      }
      numInst += 1
    }
    val acc = numCorrect.toFloat / numInst
    logger.info(s"Final accuracy = $acc")
    assert(acc > 0.92)
  }
}
