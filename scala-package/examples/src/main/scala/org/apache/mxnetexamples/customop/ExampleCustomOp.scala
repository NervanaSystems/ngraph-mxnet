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

package org.apache.mxnetexamples.customop

import org.apache.mxnet.Callback.Speedometer
import org.apache.mxnet.DType.DType
import org.apache.mxnet.{Accuracy, Context, CustomOp, CustomOpProp, NDArray, NDArrayCollector, Operator, Shape, Symbol, Xavier}
import org.apache.mxnet.optimizer.RMSProp
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * Example of CustomOp
  */
object ExampleCustomOp {
  private val logger = LoggerFactory.getLogger(classOf[ExampleCustomOp])

  class Softmax(_param: Map[String, String]) extends CustomOp {

    override def forward(sTrain: Boolean, req: Array[String], inData: Array[NDArray],
                         outData: Array[NDArray], aux: Array[NDArray]): Unit = {
      val xShape = inData(0).shape
      val x = inData(0).toArray.grouped(xShape(1)).toArray
      val yArr = x.map { it =>
        val max = it.max
        val tmp = it.map(e => Math.exp(e.toDouble - max).toFloat)
        val sum = tmp.sum
        tmp.map(_ / sum)
      }.flatten
      val y = NDArray.empty(xShape, outData(0).context)
      y.set(yArr)
      this.assign(outData(0), req(0), y)
      y.dispose()
    }

    override def backward(req: Array[String], outGrad: Array[NDArray],
                          inData: Array[NDArray], outData: Array[NDArray],
                          inGrad: Array[NDArray], aux: Array[NDArray]): Unit = {
      val l = inData(1).toArray.map(_.toInt)
      val oShape = outData(0).shape
      val yArr = outData(0).toArray.grouped(oShape(1)).toArray
      l.indices.foreach { i =>
        yArr(i)(l(i)) -= 1.0f
      }
      val y = NDArray.empty(oShape, inGrad(0).context)
      y.set(yArr.flatten)
      this.assign(inGrad(0), req(0), y)
      y.dispose()
    }
  }

  class SoftmaxProp(needTopGrad: Boolean = false)
    extends CustomOpProp(needTopGrad) {

    override def listArguments(): Array[String] = Array("data", "label")

    override def listOutputs(): Array[String] = Array("output")

    override def inferShape(inShape: Array[Shape]):
    (Array[Shape], Array[Shape], Array[Shape]) = {
      val dataShape = inShape(0)
      val labelShape = Shape(dataShape(0))
      val outputShape = dataShape
      (Array(dataShape, labelShape), Array(outputShape), null)
    }

    override def inferType(inType: Array[DType]):
    (Array[DType], Array[DType], Array[DType]) = {
      (inType, inType.take(1), null)
    }

    override def createOperator(ctx: String, inShapes: Array[Array[Int]],
                                inDtypes: Array[Int]): CustomOp = new Softmax(this.kwargs)
  }

  Operator.register("softmax", new SoftmaxProp)

  def test(dataPath : String, ctx : Context) : Float = {
    val data = Symbol.Variable("data")
    val label = Symbol.Variable("label")
    val fc1 = Symbol.api.FullyConnected(data = Some(data), num_hidden = 128, name = "fc1")
    val act1 = Symbol.api.Activation (data = Some(fc1), "relu", name = "relu")
    val fc2 = Symbol.api.FullyConnected(Some(act1), None, None, 64, name = "fc2")
    val act2 = Symbol.api.Activation(data = Some(fc2), "relu", name = "relu2")
    val fc3 = Symbol.api.FullyConnected(Some(act2), None, None, 10, name = "fc3")
    val kwargs = mutable.Map[String, Any]("label" -> label, "data" -> fc3)
    val mlp = Symbol.api.Custom(op_type = "softmax", name = "softmax", kwargs = kwargs)

    val (trainIter, testIter) =
      Data.mnistIterator(dataPath, batchSize = 100, inputShape = Shape(784))

    val datasAndLabels = trainIter.provideData ++ trainIter.provideLabel
    val (argShapes, outputShapes, auxShapes) = mlp.inferShape(datasAndLabels)

    val initializer = new Xavier(factorType = "in", magnitude = 2.34f)
    val argNames = mlp.listArguments()
    val argDict = argNames.zip(argShapes.map(s => NDArray.empty(s, ctx))).toMap

    val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
      !datasAndLabels.contains(name)
    }.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap

    argDict.foreach { case (name, ndArray) =>
      if (!datasAndLabels.contains(name)) {
        initializer.initWeight(name, ndArray)
      }
    }

    val executor = mlp.bind(ctx, argDict, gradDict)
    val lr = 0.001f
    val opt = new RMSProp(learningRate = lr, wd = 0.00001f)
    val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
      (idx, name, grad, opt.createState(idx, argDict(name)))
    }

    val evalMetric = new Accuracy
    val batchEndCallback = new Speedometer(100, 100)
    val numEpoch = 10
    var validationAcc = 0.0f

    for (epoch <- 0 until numEpoch) {
      val tic = System.currentTimeMillis
      evalMetric.reset()
      var nBatch = 0
      var epochDone = false
      NDArrayCollector.auto().withScope {
        trainIter.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && trainIter.hasNext) {
            val dataBatch = trainIter.next()
            argDict("data").set(dataBatch.data(0))
            argDict("label").set(dataBatch.label(0))
            executor.forward(isTrain = true)
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }
            evalMetric.update(dataBatch.label, executor.outputs)
            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            trainIter.reset()
          }
          epochDone = true
        }
        val (name, value) = evalMetric.get
        name.zip(value).foreach { case (n, v) =>
          logger.info(s"Epoch[$epoch] Train-accuracy=$v")
        }
        val toc = System.currentTimeMillis
        logger.info(s"Epoch[$epoch] Time cost=${toc - tic}")

        evalMetric.reset()
        testIter.reset()
        while (testIter.hasNext) {
          val evalBatch = testIter.next()
          argDict("data").set(evalBatch.data(0))
          argDict("label").set(evalBatch.label(0))
          executor.forward(isTrain = true)
          evalMetric.update(evalBatch.label, executor.outputs)
          evalBatch.dispose()
        }
        val (names, values) = evalMetric.get
        names.zip(values).foreach { case (n, v) =>
          logger.info(s"Epoch[$epoch] Validation-accuracy=$v")
          validationAcc = Math.max(validationAcc, v)
        }
      }
    }
    executor.dispose()
    validationAcc
  }

  def main(args: Array[String]): Unit = {
    val leop = new ExampleCustomOp
    val parser: CmdLineParser = new CmdLineParser(leop)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(leop.dataPath != null)

      val ctx = if (leop.gpu >= 0) Context.gpu(0) else Context.cpu()

      val dataName = Array("data")
      val labelName = Array("softmax_label")
      test(leop.dataPath, ctx)

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class ExampleCustomOp {
  @Option(name = "--data-path", usage = "the mnist data path")
  private val dataPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
}
