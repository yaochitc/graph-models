package io.yaochi.graph.algorithm.base

import com.tencent.angel.ml.math2.vector.IntFloatVector
import it.unimi.dsi.fastutil.longs.LongArrayList

import scala.collection.mutable.ArrayBuffer

class NodeFeaturePartition(val index: Int,
                           val keys: Array[Long],
                           val features: Array[IntFloatVector]) extends Serializable {
  def init(model: GNNPSModel, numBatch: Int): Unit =
    model.initNodeFeatures(keys, features, numBatch)
}

object NodeFeaturePartition {
  def apply(index: Int, iterator: Iterator[(Long, IntFloatVector)]): NodeFeaturePartition = {
    val keys = new LongArrayList()
    val features = new ArrayBuffer[IntFloatVector]()
    while (iterator.hasNext) {
      val entry = iterator.next()
      keys.add(entry._1)
      features.append(entry._2)
    }
    new NodeFeaturePartition(index, keys.toLongArray, features.toArray)
  }
}