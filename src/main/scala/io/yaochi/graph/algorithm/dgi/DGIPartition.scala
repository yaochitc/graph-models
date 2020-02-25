package io.yaochi.graph.algorithm.dgi

import io.yaochi.graph.algorithm.base._
import io.yaochi.graph.optim.AsyncOptim
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

class DGIPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   useSecondOrder: Boolean) extends
  GNNPartition[DGIPSModel, DGIModel](index, keys, indptr, neighbors) {

  override def trainEpoch(curEpoch: Int,
                          batchSize: Int,
                          model: DGIModel,
                          psModel: DGIPSModel,
                          featureDim: Int,
                          optim: AsyncOptim,
                          numSample: Int): (Double, Long) = {
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val batchIterator = keys.indices.sliding(batchSize, batchSize)
    var lossSum = 0.0
    while (batchIterator.hasNext) {
      val batch = batchIterator.next().toArray
      val loss = trainBatch(batch, model, psModel, featureDim, optim, numSample,
        srcs, dsts, batchKeys, index)
      lossSum += loss * batch.length
      srcs.clear()
      dsts.clear()
      batchKeys.clear()
      index.clear()
    }

    (lossSum, keys.length)
  }

  def trainBatch(batchIdx: Array[Int],
                 model: DGIModel,
                 psModel: DGIPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 srcs: LongArrayList,
                 dsts: LongArrayList,
                 batchKeys: LongOpenHashSet,
                 index: Long2IntOpenHashMap): Double = {

    for (idx <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
    }

    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx,
      keys, indptr, neighbors, srcs, dsts, batchKeys,
      index, numSample, psModel, useSecondOrder)

    val posx = MakeFeature.makeFeatures(index, featureDim, psModel)
    val negx = MakeFeature.sampleFeatures(index.size(), featureDim, psModel)
    assert(posx.length == negx.length)
    val weights = psModel.readWeights()

    val loss = model.backward(batchIdx.length, posx, negx,
      featureDim, first, second, weights)

    psModel.step(weights, optim)
    loss
  }
}


object DGIPartition {
  def apply(adjPartition: GraphAdjPartition,
            useSecondOrder: Boolean): DGIPartition = {
    new DGIPartition(adjPartition.index,
      adjPartition.keys,
      adjPartition.indptr,
      adjPartition.neighbours,
      useSecondOrder)
  }
}
