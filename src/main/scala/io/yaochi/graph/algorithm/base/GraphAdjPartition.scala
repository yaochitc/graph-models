package io.yaochi.graph.algorithm.base

import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.LongArrayList

class GraphAdjPartition(index: Int,
                        keys: Array[Long],
                        indptr: Array[Int],
                        neighbours: Array[Long]) extends Serializable {

  def apply(model: GNNPSModel, numBatch: Int): Unit = {
    // init adjacent table on servers
    model.initNeighbors(keys, indptr, neighbours, numBatch)
  }
}

object GraphAdjPartition {
  def apply(index: Int, iterator: Iterator[(Long, Iterable[Long])]): GraphAdjPartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbours = new LongArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      ns.foreach(n => neighbours.add(n))
      indptr.add(neighbours.size())
      keys.add(node)
    }

    new GraphAdjPartition(index, keys.toLongArray,
      indptr.toIntArray, neighbours.toLongArray)
  }
}