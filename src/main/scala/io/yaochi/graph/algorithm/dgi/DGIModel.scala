package io.yaochi.graph.algorithm.dgi

import io.yaochi.graph.algorithm.base.GNNModel

class DGIModel(inputDim: Int,
               hiddenDim: Int) extends GNNModel {

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
            hiddenDim: Int): DGIModel = new DGIModel(inputDim, hiddenDim)
}
