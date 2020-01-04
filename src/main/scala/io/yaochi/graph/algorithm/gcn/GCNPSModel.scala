package io.yaochi.graph.algorithm.gcn

import java.util.{ArrayList => JArrayList}

import com.tencent.angel.graph.data.Node
import com.tencent.angel.ml.math2.VFactory
import com.tencent.angel.ml.math2.vector.IntFloatVector
import com.tencent.angel.ml.matrix.psf.update.XavierUniform
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.spark.ml.util.LoadBalancePartitioner
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import io.yaochi.graph.algorithm.base.GNNPSModel
import io.yaochi.graph.optim.AsyncOptim
import org.apache.spark.rdd.RDD

class GCNPSModel(graph: PSMatrix,
                 labels: PSVector,
                 val weights: PSVector) extends GNNPSModel(graph, labels) {

  override def initialize(): Unit = {
    weights.psfUpdate(new XavierUniform(weights.poolId, 0, 1, 1.0, 1, weights.dimension)).get()
  }

  def readWeights(): Array[Float] =
    weights.pull().asInstanceOf[IntFloatVector].getStorage.getValues

  def setWeights(values: Array[Float]): Unit = {
    val update = VFactory.denseFloatVector(values)
    weights.update(update)
  }

  def step(grads: Array[Float], optim: AsyncOptim): Unit = {
    val update = VFactory.denseFloatVector(grads)
    optim.asycUpdate(weights, 1, update).get()
  }

}

object GCNPSModel {
  def apply(minId: Long, maxId: Long, weightSize: Int, optim: AsyncOptim,
            index: RDD[Long], psNumPartition: Int,
            useBalancePartition: Boolean = false): GCNPSModel = {
    val graph = new MatrixContext("graph", 1, minId, maxId)
    graph.setRowType(RowType.T_ANY_LONGKEY_SPARSE)
    graph.setValueType(classOf[Node])

    val labels = new MatrixContext("labels", 1, minId, maxId)
    labels.setRowType(RowType.T_FLOAT_SPARSE_LONGKEY)

    if (useBalancePartition)
      LoadBalancePartitioner.partition(index, maxId, psNumPartition, graph)

    val weights = new MatrixContext("weights", optim.getNumSlots(), weightSize)
    weights.setRowType(RowType.T_FLOAT_DENSE)
    weights.setPartitionerClass(classOf[ColumnRangePartitioner])

    val list = new JArrayList[MatrixContext]()
    list.add(graph)
    list.add(weights)
    list.add(labels)

    PSAgentContext.get().getMasterClient.createMatrices(list, 10000L)
    val graphId = PSAgentContext.get().getMasterClient.getMatrix("graph").getId
    val weightsId = PSAgentContext.get().getMasterClient.getMatrix("weights").getId
    val labelsId = PSAgentContext.get().getMasterClient.getMatrix("labels").getId

    new GCNPSModel(new PSMatrixImpl(graphId, 1, maxId, graph.getRowType),
      new PSVectorImpl(labelsId, 0, maxId, labels.getRowType),
      new PSVectorImpl(weightsId, 0, weights.getColNum, weights.getRowType)
    )
  }
}
