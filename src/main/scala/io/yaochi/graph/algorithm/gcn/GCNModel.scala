package io.yaochi.graph.algorithm.gcn

import io.yaochi.graph.algorithm.base.GNNModel

class GCNModel(hiddenDim: Int,
               numClasses: Int) extends GNNModel {

  def forward(batchSize: Int,
              x: Array[Float],
              featureDim: Int,
              firstEdgeIndex: Array[Long],
              secondEdgeIndex: Array[Long],
              weights: Array[Float]): Array[Float] = {
    val numFirstOrderNodes = batchSize
    val firstEdgeNorms = GCNModel.calcNorms(firstEdgeIndex, numFirstOrderNodes)

    val numSecondOrderNodes = x.length / featureDim
    val secondEdgeNorms = GCNModel.calcNorms(secondEdgeIndex, numSecondOrderNodes)

    val secondOrderEncoder = GCNEncoder(numSecondOrderNodes, featureDim, hiddenDim, weights, reshape = true)
    val offset = secondOrderEncoder.parameterSize

    val firstOrderEncoder = GCNEncoder(numFirstOrderNodes, hiddenDim, numClasses, weights, offset)

    null
  }

  def backward(batchSize: Int,
               x: Array[Float],
               featureDim: Int,
               firstEdgeIndex: Array[Long],
               secondEdgeIndex: Array[Long],
               weights: Array[Float],
               targets: Array[Long]): Float = {
    val numFirstOrderNodes = batchSize
    val firstEdgeNorms = GCNModel.calcNorms(firstEdgeIndex, numFirstOrderNodes)

    val numSecondOrderNodes = x.length / featureDim
    val secondEdgeNorms = GCNModel.calcNorms(secondEdgeIndex, numSecondOrderNodes)

    0f
  }
}

object GCNModel {
  def apply(hiddenDim: Int,
            numClasses: Int): GCNModel = new GCNModel(hiddenDim, numClasses)


  def calcNorms(edgeIndex: Array[Long], numNodes: Int): Array[Float] = {
    val size = edgeIndex.length / 2
    val norms = Array.ofDim[Float](numNodes)
    for (i <- 0 until size) {
      val src = edgeIndex(i).toInt
      norms(src) += 1
    }

    for (i <- 0 until numNodes) {
      if (norms(i) > 0) {
        norms(i) = Math.pow(norms(i), -0.5).toFloat
      }
    }

    norms
  }

}
