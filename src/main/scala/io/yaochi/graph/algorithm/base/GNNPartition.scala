package io.yaochi.graph.algorithm.base

import io.yaochi.graph.optim.AsyncOptim

abstract class GNNPartition[PSModel <: GNNPSModel](index: Int,
                                                   keys: Array[Long],
                                                   indptr: Array[Int],
                                                   neighbors: Array[Long],
                                                   useSecondOrder: Boolean) extends Serializable {

  def numNodes: Long = keys.length

  def numEdges: Long = neighbors.length

  def trainEpoch(curEpoch: Int,
                 batchSize: Int,
                 model: PSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int): (Double, Long)

}
