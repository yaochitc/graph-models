package io.yaochi.graph.algorithm.gcn

import io.yaochi.graph.algorithm.base.GNNModel

class GCNModel(inputDim: Int,
               hiddenDim: Int,
               numClasses: Int) extends GNNModel {

  def forward(batchSize: Int,
              x: Array[Float],
              featureDim: Int,
              firstEdgeIndex: Array[Long],
              secondEdgeIndex: Array[Long],
              weights: Array[Float]): Array[Float] = {
    null
  }

  def backward(batchSize: Int,
               x: Array[Float],
               featureDim: Int,
               firstEdgeIndex: Array[Long],
               secondEdgeIndex: Array[Long],
               weights: Array[Float],
               targets: Array[Long]): Float = {
    0f
  }
}

object GCNModel {
  def apply(inputDim: Int,
            hiddenDim: Int,
            numClasses: Int): GCNModel = new GCNModel(inputDim, hiddenDim, numClasses)
}
