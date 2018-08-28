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

package org.apache.mxnetexamples.cnntextclassification

import org.apache.mxnet.optimizer.RMSProp
import org.apache.mxnet.{Context, Executor, Model, NDArray, NDArrayCollector, Optimizer, Shape, Symbol, Uniform}
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.util.Random

/**
  * An Implementation of the paper
  * Convolutional Neural Networks for Sentence Classification
  */
object CNNTextClassification {

  private val logger = LoggerFactory.getLogger(classOf[CNNTextClassification])

  case class CNNModel(cnnExec: Executor, symbol: Symbol, data: NDArray, label: NDArray,
                      argsDict: Map[String, NDArray], gradDict: Map[String, NDArray])

  def makeTextCNN(sentenceSize: Int, numEmbed: Int, batchSize: Int,
                  numLabel: Int = 2, filterList: Array[Int] = Array(3, 4, 5), numFilter: Int = 100,
                  dropout: Float = 0.5f): Symbol = {

    val inputX = Symbol.Variable("data")
    val inputY = Symbol.Variable("softmax_label")
    val polledOutputs = filterList.map { filterSize =>
      val conv = Symbol.api.Convolution(data = Some(inputX),
        kernel = new Shape(filterSize, numEmbed), num_filter = numFilter)
      val relu = Symbol.api.Activation(data = Some(conv), act_type = "relu")
      val pool = Symbol.api.Pooling(data = Some(relu), pool_type = Some("max"),
        kernel = Some(new Shape(sentenceSize - filterSize + 1, 1)), stride = Some(new Shape(1, 1)))
      relu.dispose()
      conv.dispose()
      pool
    }

    val totalFilters = numFilter * filterList.length
    // val concat = Symbol.Concat()(polledOutputs: _*)(Map("dim" -> 1))
    val concat = Symbol.api.concat(data = polledOutputs,
      num_args = polledOutputs.length, dim = Some(1))
    val hPool = Symbol.api.reshape(data = Some(concat),
      target_shape = Some(new Shape(batchSize, totalFilters)))

    val hDrop = {
      if (dropout > 0f) Symbol.api.Dropout(data = Some(hPool), p = Some(dropout))
      else hPool
    }

    val fc = Symbol.api.FullyConnected(data = Some(hDrop), num_hidden = numLabel)
    val sm = Symbol.api.SoftmaxOutput(data = Some(fc), label = Some(inputY))
    fc.dispose()
    hDrop.dispose()
    hPool.dispose()
    concat.dispose()
    polledOutputs.foreach(_.dispose())
    inputX.dispose()
    inputY.dispose()
    sm
  }

  def setupCnnModel(ctx: Context, batchSize: Int, sentenceSize: Int, numEmbed: Int,
      numLabel: Int = 2, numFilter: Int = 100, filterList: Array[Int ] = Array(3, 4, 5),
      dropout: Float = 0.0f): CNNModel = {

    val cnn = makeTextCNN(sentenceSize, numEmbed, batchSize,
      numLabel, filterList, numFilter, dropout)
    val argNames = cnn.listArguments()
    val auxNames = cnn.listAuxiliaryStates()

    val (argShapes, outShapes, auxShapes) = cnn.inferShape(
      Map("data" -> Shape(batchSize, 1, sentenceSize, numEmbed)))
    val argsDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
    val argsGradDict = argNames.zip(argShapes)
      .filter(x => x._1 != "softmax_label" && x._1 != "data")
      .map(x => x._1 -> NDArray.zeros(x._2, ctx)).toMap
    val auxDict = auxNames.zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap
    val cnnExec = cnn.bind(ctx, argsDict, argsGradDict, "add", auxDict, null, null)

    val data = argsDict("data")
    val label = argsDict("softmax_label")
    CNNModel(cnnExec, cnn, data, label, argsDict, argsGradDict)
  }

  def trainCNN(model: CNNModel, trainBatches: Array[Array[Array[Float]]],
               trainLabels: Array[Float], devBatches: Array[Array[Array[Float]]],
               devLabels: Array[Float], batchSize: Int, saveModelPath: String,
               learningRate: Float = 0.001f): Float = {
    val maxGradNorm = 0.5f
    val epoch = 10
    val initializer = new Uniform(0.1f)
    val opt = new RMSProp(learningRate)
    val updater = Optimizer.getUpdater(opt)
    var start = 0L
    var end = 0L
    var numCorrect = 0f
    var numTotal = 0f
    var factor = 0.5f
    var maxAccuracy = -1f
    var updateRate = 0

    val paramBlocks = model.symbol.listArguments()
      .filter(x => x != "data" && x != "softmax_label")
      .zipWithIndex.map { x =>
      initializer(x._1, model.gradDict(x._1))
      val state = opt.createState(x._2, model.argsDict(x._1))
      (x._2, model.argsDict(x._1), model.gradDict(x._1), state, x._1)
    }.toArray
    var devAcc = 0.0f
    for (iter <- 0 until epoch) {
      start = System.currentTimeMillis()
      numCorrect = 0f
      numTotal = 0f
      updateRate = 0

      NDArrayCollector.auto().withScope {
        for (begin <- 0 until trainBatches.length by batchSize) {
          val (batchD, batchL) = {
            if (begin + batchSize <= trainBatches.length) {
              val datas = trainBatches.drop(begin).take(batchSize)
              val labels = trainLabels.drop(begin).take(batchSize)
              (datas, labels)
            } else {
              val right = (begin + batchSize) - trainBatches.length
              val left = trainBatches.length - begin
              val datas = trainBatches.drop(begin).take(left) ++ trainBatches.take(right)
              val labels = trainLabels.drop(begin).take(left) ++ trainLabels.take(right)
              (datas, labels)
            }
          }
          numTotal += batchSize
          model.data.set(batchD.flatten.flatten)
          model.label.set(batchL)

          model.cnnExec.forward(isTrain = true)
          model.cnnExec.backward()

          val tmpCorrect = {
            val predLabel = NDArray.api.argmax_channel(model.cnnExec.outputs(0))
            val result = predLabel.toArray.zip(batchL).map { predLabel =>
              if (predLabel._1 == predLabel._2) 1
              else 0
            }.sum.toFloat
            predLabel.dispose()
            result
          }

          numCorrect = numCorrect + tmpCorrect
          val norm = Math.sqrt(paramBlocks.map { case (idx, weight, grad, state, name) =>
            val temp = NDArray.api.norm(grad / batchSize).disposeDepsExcept(grad)
            val l2Norm = temp.toScalar
            temp.dispose()
            l2Norm * l2Norm
          }.sum).toFloat

          if (updateRate % 2 == 0) {
            paramBlocks.foreach { case (idx, weight, grad, state, name) =>
              if (norm > maxGradNorm) {
                grad.set(grad.toArray.map(_ * (maxGradNorm / norm)))
                opt.update(idx, weight, grad, state)
              }
              else opt.update(idx, weight, grad, state)
              grad.set(0f)
            }
          }
          updateRate = updateRate + 1
        }
      }

      // decay learning rate
      if (iter % 50 == 0 && iter > 0) {
        factor *= 0.5f
        opt.setLrScale(paramBlocks.map(_._1 -> factor).toMap)
        logger.info(s"reset learning to ${opt.learningRate * factor}")
      }
      // end of training loop
      end = System.currentTimeMillis()
      logger.info(s"Iter $iter Train: Time: ${(end - start) / 1000}," +
        s"Training Accuracy: ${numCorrect / numTotal * 100}%")

      // eval on dev set
      numCorrect = 0f
      numTotal = 0f
      for (begin <- 0 until devBatches.length by batchSize) {
        if (begin + batchSize <= devBatches.length) {
          numTotal += batchSize
          val (batchD, batchL) = {
            val datas = devBatches.drop(begin).take(batchSize)
            val labels = devLabels.drop(begin).take(batchSize)
            (datas, labels)
          }

          model.data.set(batchD.flatten.flatten)
          model.label.set(batchL)

          model.cnnExec.forward(isTrain = false)

          val tmpCorrect = {
            val predLabel = NDArray.api.argmax_channel(model.cnnExec.outputs(0))
            val result = predLabel.toArray.zip(batchL).map { predLabel =>
              if (predLabel._1 == predLabel._2) 1
              else 0
            }.sum.toFloat
            predLabel.dispose()
            result
          }
          numCorrect = numCorrect + tmpCorrect
        }
      }
      devAcc = numCorrect / numTotal
      logger.info(s"Dev Accuracy so far: ${devAcc * 100}%")
      if (devAcc > maxAccuracy) {
        maxAccuracy = devAcc
        Model.saveCheckpoint(s"$saveModelPath/cnn-text-dev-acc-$maxAccuracy",
          iter, model.symbol, model.cnnExec.argDict, model.cnnExec.auxDict)
        logger.info(s"max accuracy on dev so far: ${maxAccuracy  * 100}%")
      }
    }
    devAcc
  }

  def test(w2vFilePath : String, mrDatasetPath: String,
           ctx : Context, saveModelPath: String) : Float = {
    val output = NDArrayCollector.auto().withScope {
      val (numEmbed, word2vec) = DataHelper.loadGoogleModel(w2vFilePath)
      val (datas, labels) = DataHelper.loadMSDataWithWord2vec(
        mrDatasetPath, numEmbed, word2vec)
      // randomly shuffle data
      val randIdx = Random.shuffle((0 until datas.length).toList)
      // split train/dev set
      val (trainDats, devDatas) = {
        val train = randIdx.dropRight(1000).map(datas(_)).toArray
        val dev = randIdx.takeRight(1000).map(datas(_)).toArray
        (train, dev)
      }
      val (trainLabels, devLabels) = {
        val train = randIdx.dropRight(1000).map(labels(_)).toArray
        val dev = randIdx.takeRight(1000).map(labels(_)).toArray
        (train, dev)
      }
      // reshpae for convolution input
      val sentenceSize = datas(0).length
      val batchSize = 100
      val lr = 0.001f
      val cnnModel = setupCnnModel(ctx, batchSize, sentenceSize, numEmbed)
      val result = trainCNN(cnnModel, trainDats, trainLabels, devDatas, devLabels, batchSize,
        saveModelPath, learningRate = lr)
      result
    }
    output
  }

  def main(args: Array[String]): Unit = {
    val exon = new CNNTextClassification
    val parser: CmdLineParser = new CmdLineParser(exon)
    try {
      parser.parseArgument(args.toList.asJava)

      logger.info("Loading data...")
      val (numEmbed, word2vec) =
        if (exon.w2vFormatBin == 1) DataHelper.loadGoogleModel(exon.w2vFilePath)
        else DataHelper.loadPretrainedWord2vec(exon.w2vFilePath)
      val (datas, labels) = DataHelper.loadMSDataWithWord2vec(
        exon.mrDatasetPath, numEmbed, word2vec)

      // randomly shuffle data
      val randIdx = Random.shuffle((0 until datas.length).toList)
      // split train/dev set
      val (trainDats, devDatas) = {
        val train = randIdx.dropRight(1000).map(datas(_)).toArray
        val dev = randIdx.takeRight(1000).map(datas(_)).toArray
        (train, dev)
      }
      val (trainLabels, devLabels) = {
        val train = randIdx.dropRight(1000).map(labels(_)).toArray
        val dev = randIdx.takeRight(1000).map(labels(_)).toArray
        (train, dev)
      }

      // reshpae for convolution input
      val sentenceSize = datas(0).length
      val ctx = if (exon.gpu == -1) Context.cpu() else Context.gpu(exon.gpu)

      val cnnModel = setupCnnModel(ctx, exon.batchSize, sentenceSize, numEmbed)
      trainCNN(cnnModel, trainDats, trainLabels, devDatas, devLabels, exon.batchSize,
        exon.saveModelPath, learningRate = exon.lr)

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class CNNTextClassification {
  @Option(name = "--lr", usage = "the initial learning rate")
  private val lr: Float = 0.001f
  @Option(name = "--batch-size", usage = "the batch size")
  private val batchSize: Int = 100
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
  @Option(name = "--w2v-format-bin", usage = "does the word2vec file format is binary")
  private val w2vFormatBin: Int = 0
  @Option(name = "--mr-dataset-path", usage = "the MR polarity dataset path")
  private val mrDatasetPath: String = ""
  @Option(name = "--w2v-file-path", usage = "the word2vec file path")
  private val w2vFilePath: String = ""
  @Option(name = "--save-model-path", usage = "the model saving path")
  private val saveModelPath: String = ""
}
