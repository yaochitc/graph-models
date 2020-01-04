package io.yaochi.graph.algorithm.base

import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.vector.LongFloatVector

class NodeLabelPartition(val index: Int,
                         val labels: LongFloatVector) extends Serializable {
  def init(model: SupervisedGNNPSModel): Unit =
    model.setLabels(labels)
}

object NodeLabelPartition {
  def apply(index: Int, iterator: Iterator[(Long, Float)], dim: Long): NodeLabelPartition = {
    val labels = VFactory.sparseLongKeyFloatVector(dim)
    while (iterator.hasNext) {
      val entry = iterator.next()
      labels.set(entry._1, entry._2)
    }
    new NodeLabelPartition(index, labels)
  }
}