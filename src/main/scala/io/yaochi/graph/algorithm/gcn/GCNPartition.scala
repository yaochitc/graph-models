package io.yaochi.graph.algorithm.gcn

import io.yaochi.graph.algorithm.base._
import io.yaochi.graph.optim.AsyncOptim
import it.unimi.dsi.fastutil.longs.{Long2IntOpenHashMap, LongArrayList, LongOpenHashSet}

class GCNPartition(index: Int,
                   keys: Array[Long],
                   indptr: Array[Int],
                   neighbors: Array[Long],
                   trainIdx: Array[Int],
                   trainLabels: Array[Float],
                   testIdx: Array[Int],
                   testLabels: Array[Float]) extends
  GNNPartition[GCNPSModel](index, keys, indptr, neighbors, true) {

  def getTrainTestSize: (Int, Int) = (trainIdx.length, testIdx.length)

  override def trainEpoch(curEpoch: Int,
                          batchSize: Int,
                          model: GCNPSModel,
                          featureDim: Int,
                          optim: AsyncOptim,
                          numSample: Int): (Double, Long) = {
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val batchIterator = trainIdx.zip(trainLabels).sliding(batchSize, batchSize)
    var lossSum = 0.0
    var numRight: Long = 0

    while (batchIterator.hasNext) {
      val batch = batchIterator.next()
      val (loss, right) = trainBatch(batch, model, featureDim,
        optim, numSample, srcs, dsts, batchKeys, index)
      lossSum += loss * batch.length
      numRight += right
      srcs.clear()
      dsts.clear()
      batchKeys.clear()
      index.clear()
    }

    (lossSum, numRight)
  }

  def trainBatch(batchIdx: Array[(Int, Float)],
                 model: GCNPSModel,
                 featureDim: Int,
                 optim: AsyncOptim,
                 numSample: Int,
                 srcs: LongArrayList,
                 dsts: LongArrayList,
                 batchKeys: LongOpenHashSet,
                 index: Long2IntOpenHashMap): (Double, Long) = {
    val targets = new Array[Long](batchIdx.length)
    var k = 0
    for ((idx, label) <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
      targets(k) = label.toLong
      k += 1
    }

    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx.map(f => f._1),
      keys, indptr, neighbors, srcs, dsts,
      batchKeys, index, numSample, model, true)
    val x = MakeFeature.makeFeatures(index, featureDim, model)

    val weights = model.readWeights()
    val loss = 0
    model.step(weights, optim)

    var right: Long = 0
    (loss, right)
  }

  def predictEpoch(curEpoch: Int,
                   batchSize: Int,
                   model: GCNPSModel,
                   featureDim: Int,
                   numSample: Int): Long = {
    val index = new Long2IntOpenHashMap()
    val srcs = new LongArrayList()
    val dsts = new LongArrayList()
    val batchKeys = new LongOpenHashSet()
    val batchIterator = testIdx.zip(testLabels).sliding(batchSize, batchSize)
    var numRight: Int = 0
    val weights = model.readWeights()

    while (batchIterator.hasNext) {
      val batch = batchIterator.next()
      val right = predictBatch(batch, model, featureDim, numSample,
        srcs, dsts, batchKeys, index, weights)
      numRight += right
      srcs.clear()
      dsts.clear()
      batchKeys.clear()
      index.clear()
    }

    numRight
  }

  def predictBatch(batchIdx: Array[(Int, Float)],
                   model: GNNPSModel,
                   featureDim: Int,
                   numSample: Int,
                   srcs: LongArrayList,
                   dsts: LongArrayList,
                   batchKeys: LongOpenHashSet,
                   index: Long2IntOpenHashMap,
                   weights: Array[Float]): Int = {
    val targets = new Array[Long](batchIdx.length)
    var k = 0
    for ((idx, label) <- batchIdx) {
      batchKeys.add(keys(idx))
      index.put(keys(idx), index.size())
      targets(k) = label.toLong
      k += 1
    }

    val (first, second) = MakeEdgeIndex.makeEdgeIndex(batchIdx.map(f => f._1),
      keys, indptr, neighbors, srcs, dsts,
      batchKeys, index, numSample, model, true)
    val x = MakeFeature.makeFeatures(index, featureDim, model)
    var right = 0
    right
  }
}

object GCNPartition {
  def apply(adjPartition: GraphAdjPartition,
            model: GCNPSModel,
            testRatio: Float): GCNPartition = {
    val keys = adjPartition.keys
    val myLabels = model.readLabels2(keys)
    val it = myLabels.getStorage.entryIterator()
    val size = myLabels.size().toInt
    val idxHasLabels = new Array[Int](size)
    val labels = new Array[Float](size)
    val position = new Long2IntOpenHashMap(keys.length)
    for (idx <- keys.indices)
      position.put(keys(idx), idx)
    var idx = 0
    while (it.hasNext) {
      val entry = it.next()
      val (node, label) = (entry.getLongKey, entry.getFloatValue)
      idxHasLabels(idx) = position.get(node)
      labels(idx) = label
      idx += 1
    }

    val splitPoint = (size * (1 - testRatio)).toInt
    val (trainIdx, testIdx) = idxHasLabels.splitAt(splitPoint)
    val (trainLabels, testLabels) = labels.splitAt(splitPoint)
    new GCNPartition(adjPartition.index,
      adjPartition.keys,
      adjPartition.indptr,
      adjPartition.neighbours,
      trainIdx,
      trainLabels,
      testIdx,
      testLabels)
  }
}

