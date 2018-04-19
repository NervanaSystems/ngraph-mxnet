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

package org.apache.mxnet.spark.utils

import javax.imageio.ImageIO

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.input._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
 * Img2Vector tools could convert imgae directory into Vectorized RDD,for example:
 * Images stored in hdfs://namenode:9000/user/xxx/images/
 * val sc = new SparkContext(conf)
 * val imagesArrayRDD = Img2Vector.getRGBArray(sc, "hdfs://namenode:9000/user/xxx/images/")
 * val imagesVectorRDD = Img2Vector.getRGBVector(sc, "hdfs://namenode:9000/user/xxx/images/")
 * @author Yuance.Li
 */
object Img2Vector{
  def getImgRGB(PDS: PortableDataStream, fullcolor: Boolean): Array[Double] = {
    val img = ImageIO.read(PDS.open())
    val R = ArrayBuffer[Double]()
    val G = ArrayBuffer[Double]()
    val B = ArrayBuffer[Double]()
    val RGB = ArrayBuffer[Double]()
    val w = img.getWidth
    val h = img.getHeight
    if (fullcolor) {
      for (x <- 0 until w){
        for (y <- 0 until h) {
          val color = img.getRGB(w - x - 1, y) & 0xffffff
          R += (color & 0xff0000) / 65536
          G += (color & 0xff00) / 256
          B += (color & 0xff)
        }
      }
      RGB ++= R ++= G ++= B
      RGB.toArray
    } else {
      for (x <- 0 until w) {
        for (y <- 0 until h){
          val color = img.getRGB(w - x - 1, y) & 0xffffff
          R += (color & 0xff0000) / 65536 * 0.3
          G += (color & 0xff00) / 256 * 0.59
          B += (color & 0xff) * 0.11
        }
      }
      val grayArr = new Array[Double](w * h)
      for (i <- 0 until w * h) {
        grayArr(i) = R(i) + G(i) + B(i)
      }
      grayArr
    }
  }

  def getRGBArray(sc: SparkContext, path: String, fullcolor: Boolean = true): RDD[Array[Double]] = {
    val rgbArray = sc.binaryFiles(path).map(_._2).map(getImgRGB(_, fullcolor))
    rgbArray
  }

  def getRGBVector(sc: SparkContext, path: String, fullcolor: Boolean = true): RDD[Vector] = {
    val rgbArray = sc.binaryFiles(path).map(_._2).map(getImgRGB(_, fullcolor))
    val rgbVector = rgbArray.map(x => Vectors.dense(x))
    rgbVector
  }
}
