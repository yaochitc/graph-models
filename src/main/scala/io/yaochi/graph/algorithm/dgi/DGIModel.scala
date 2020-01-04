package io.yaochi.graph.algorithm.dgi

import io.yaochi.graph.algorithm.base.GNNModel

class DGIModel(inputDim: Int,
               hiddenDim: Int,
               outputDim: Int) extends GNNModel {

  def getParameterSize: Int = {
    val conv1ParamSize = 2 * inputDim * hiddenDim + hiddenDim
    val conv2ParamSize = 2 * hiddenDim * outputDim + outputDim
    conv1ParamSize + conv2ParamSize + hiddenDim + outputDim
  }

  def backward(batchSize: Int,
               posX: Array[Float],
               negX: Array[Float],
               featureDim: Int,
               firstEdgeIndex: Array[Long],
               secondEdgeIndex: Array[Long],
               weights: Array[Float]): Float = {
    0f
  }
}

object DGIModel {
  def apply(inputDim: Int,
            hiddenDim: Int,
            outputDim: Int): DGIModel = new DGIModel(inputDim, hiddenDim, outputDim)
}
