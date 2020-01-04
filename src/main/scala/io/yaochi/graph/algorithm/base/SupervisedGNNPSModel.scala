package io.yaochi.graph.algorithm.base

import com.tencent.angel.ml.math2.vector.LongFloatVector
import com.tencent.angel.spark.ml.psf.gcn.{GetLabels, GetLabelsResult}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import com.tencent.angel.spark.util.VectorUtils

class SupervisedGNNPSModel(graph: PSMatrix,
                           val labels: PSVector) extends GNNPSModel(graph) {

  val dim: Long = labels.dimension

  // the default pull method will return keys even those not exists on servers
  def readLabels(keys: Array[Long]): LongFloatVector =
    labels.pull(keys.clone()).asInstanceOf[LongFloatVector]

  // this method will not return keys that do not exist on servers
  def readLabels2(keys: Array[Long]): LongFloatVector = {
    val func = new GetLabels(labels.poolId, keys.clone())
    labels.psfGet(func).asInstanceOf[GetLabelsResult].getVector
  }

  def setLabels(value: LongFloatVector): Unit =
    labels.update(value)

  def nnzLabels(): Long =
    VectorUtils.size(labels)
}
