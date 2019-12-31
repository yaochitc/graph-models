package io.yaochi.graph.algorithm.gcn

class GCN {
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
