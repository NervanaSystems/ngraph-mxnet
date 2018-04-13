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

package org.apache.mxnetexamples.neuralstyle.end2end

import org.apache.mxnet.Shape
import org.apache.mxnet.Context
import org.apache.mxnet.NDArray
import org.apache.mxnet.Symbol
import org.apache.mxnet.Initializer
import org.slf4j.LoggerFactory

/**
 * @author Depeng Liang
 */
object Basic {

  class PretrainedInit(prefix: String, params: Map[String, NDArray],
      verbose: Boolean = false) extends Initializer {

    private val logger = LoggerFactory.getLogger(classOf[PretrainedInit])

    private val prefixLen = prefix.length() + 1
    private val argParams = params.filter(_._1.startsWith("arg:"))
    private val auxParams = params.filter(_._1.startsWith("aux:"))
    private val argNames = argParams.keySet.map(_.substring(4))
    private val auxNames = auxParams.keySet.map(_.substring(4))

    override def initWeight(name: String, arr: NDArray): Unit = {
      val key = name.substring(prefixLen)
      if (this.argNames.contains(key)) {
        if (verbose) logger.info(s"Init $name")
        arr.set(this.argParams(s"arg:$key"))
      } else if (this.auxNames.contains(key)) {
        if (verbose) logger.info(s"Init $name")
        arr.set(this.auxParams(s"aux:$key"))
      } else {
        logger.info(s"Unknown params: $name, init with 0")
        arr.set(0f)
      }
    }

    override def initDefault(name: String, arr: NDArray): Unit = {
    }
  }

  def getStyleModule(prefix: String, dShape: Shape,
      ctx: Context, params: Map[String, NDArray]): Module = {
    val inputShape = Map(s"${prefix}_data" -> dShape)
    val (style, content) = ModelVgg19.getVggSymbol(prefix)
    val (gram, gScale) = styleGramSymbol(inputShape, style)
    val init = new PretrainedInit(prefix, params, true)
    new Module(symbol = gram, context = ctx,
                        dataShapes = Map(s"${prefix}_data" -> dShape),
                        initializer = init, forTraining = false)
  }

  def styleGramSymbol(inputShape: Map[String, Shape], style: Symbol): (Symbol, List[Int]) = {
    val (_, outputShape, _) = style.inferShape(inputShape)
    var gramList = List[Symbol]()
    var gradScale = List[Int]()
    for (i <- 0 until style.listOutputs().length) {
      val shape = outputShape(i)
      val x = Symbol.Reshape()()(Map("data" -> style.get(i),
          "shape" -> Shape(shape(1), shape(2) * shape(3))))
      // use fully connected to quickly do dot(x, x^T)
      val gram = Symbol.FullyConnected()()(Map("data" -> x, "weight" -> x,
          "no_bias" -> true, "num_hidden" -> shape(1)))
      gramList = gramList :+ gram
      gradScale = gradScale :+ (shape(1) * shape(2) * shape(3) * shape(1))
    }
    (Symbol.Group(gramList: _*), gradScale)
  }

  def getLoss(gram: Symbol, content: Symbol): (Symbol, Symbol) = {
    var gramLoss = List[Symbol]()
    for (i <- 0 until gram.listOutputs().length) {
      val gvar = Symbol.Variable(s"target_gram_$i")
      gramLoss = gramLoss :+ Symbol.sum()(Symbol.square()(gvar - gram.get(i))())()
    }
    val cvar = Symbol.Variable("target_content")
    val contentLoss = Symbol.sum()(Symbol.square()(cvar - content)())()
    (Symbol.Group(gramLoss: _*), contentLoss)
  }

  def getContentModule(prefix: String, dShape: Shape,
      ctx: Context, params: Map[String, NDArray]): Module = {
    val (_, sym) = ModelVgg19.getVggSymbol(prefix, true)
    val init = new PretrainedInit(prefix, params)
    new Module(symbol = sym, context = ctx,
                    dataShapes = Map(s"${prefix}_data" -> dShape),
                    initializer = init, forTraining = false)
  }

  def getLossModule(prefix: String, dShape: Shape,
      ctx: Context, params: Map[String, NDArray]): (Module, List[Int]) = {
    val inputShape = Map(s"${prefix}_data" -> dShape)
    val (style, content) = ModelVgg19.getVggSymbol(prefix)
    val (gram, gScale) = styleGramSymbol(inputShape, style)
    val (styleLoss, contentLoss) = getLoss(gram, content)
    val sym = Symbol.Group(styleLoss, contentLoss)
    val init = new PretrainedInit(prefix, params, true)
    val mod = new Module(symbol = sym, context = ctx,
                         dataShapes = Map(s"${prefix}_data" -> dShape),
                         initializer = init, forTraining = true,
                         inputsNeedGrad = true)
    (mod, gScale)
  }
}
