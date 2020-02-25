package io.yaochi.graph.algorithm.base

import it.unimi.dsi.fastutil.floats.FloatArrayList
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.LongArrayList

class GraphAdjPartition(val keys: Array[Long],
                        val indptr: Array[Int],
                        val neighbours: Array[Long],
                        val types: Array[Int],
                        val weights: Array[Float]
                       ) extends Serializable {

  def init(model: GNNPSModel, numBatch: Int): Int = {
    // init adjacent table on servers
    model.initNeighbors(keys, indptr, neighbours, numBatch)
    0
  }
}

object GraphAdjPartition {
  def apply(iterator: Iterator[(Long, Iterable[Edge])],
            hasType: Boolean, hasWeight: Boolean): GraphAdjPartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbours = new LongArrayList()
    val types: IntArrayList = if (hasType) new IntArrayList() else null
    val weights: FloatArrayList = if (hasWeight) new FloatArrayList() else null

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      for (n <- ns) {
        neighbours.add(n.dst)

        if (hasType) {
          types.add(n.`type`.get)
        }

        if (hasWeight) {
          weights.add(n.weight.get)
        }

      }
      indptr.add(neighbours.size())
      keys.add(node)
    }

    new GraphAdjPartition(keys.toLongArray,
      indptr.toIntArray, neighbours.toLongArray,
      if (hasType) types.toIntArray else null,
      if (hasWeight) weights.toFloatArray else null)
  }
}