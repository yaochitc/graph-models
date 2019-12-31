package io.yaochi.graph.algorithm.dgi

import io.yaochi.graph.algorithm.base._
import io.yaochi.graph.optim.AsyncOptim
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

class DGIPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   useSecondOrder: Boolean) extends
  GNNPartition[DGIPSModel](index, keys, indptr, neighbors, useSecondOrder) {

  override def trainEpoch(curEpoch: Int,
                          batchSize: Int,
                          model: DGIPSModel,
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
      val loss = trainBatch(batch, model, featureDim, optim, numSample,
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
                 model: DGIPSModel,
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
      index, numSample, model, useSecondOrder)

    val posx = MakeFeature.makeFeatures(index, featureDim, model)
    val negx = MakeFeature.sampleFeatures(index.size(), featureDim, model)
    assert(posx.length == negx.length)
    val weights = model.readWeights()

    val loss = 0

    model.step(weights, optim)
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
