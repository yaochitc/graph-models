package io.yaochi.graph.algorithm.base

import it.unimi.dsi.fastutil.floats.FloatArrayList
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.LongArrayList

class GraphAdjPartition(val keys: Array[Long],
                        val indptr: Array[Int],
                        val neighbours: Array[Long]
                       ) extends Serializable {

  def init(model: GNNPSModel, numBatch: Int): Int = {
    // init adjacent table on servers
    model.initNeighbors(keys, indptr, neighbours, numBatch)
    0
  }
}

object GraphAdjPartition {
  def apply(iterator: Iterator[(Long, Iterable[(Long, Long)])]): GraphAdjPartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbours = new LongArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = entry
      for (n <- ns) {
        neighbours.add(n._2)
      }
      indptr.add(neighbours.size())
      keys.add(node)
    }

    new GraphAdjPartition(keys.toLongArray,
      indptr.toIntArray, neighbours.toLongArray)
  }
}