package io.yaochi.graph.algorithm.dgi

import io.yaochi.graph.algorithm.base.{GNN, GraphAdjPartition}
import io.yaochi.graph.params.{HasHiddenDim, HasInputDim, HasUseSecondOrder}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

class DGI extends GNN[DGIPSModel, DGIModel]
  with HasInputDim with HasHiddenDim
  with HasUseSecondOrder {

  override def makeModel(): DGIModel = DGIModel($(inputDim), $(hiddenDim))

  override def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: DGIModel): DGIPSModel = {
    DGIPSModel.apply(minId, maxId + 1, 0, getOptimizer,
      index, $(psPartitionNum), $(useBalancePartition))
  }

  override def makeGraph(edges: RDD[(Long, Long)], model: DGIPSModel): Dataset[_] = {
    val adj = edges.groupByKey($(partitionNum))

    if ($(useSecondOrder)) {
      // if second order is required, init neighbors on PS
      adj.mapPartitionsWithIndex((index, it) =>
        Iterator(GraphAdjPartition.apply(index, it)))
        .map(_.init(model, $(numBatchInit))).reduce(_ + _)
    }

    val dgiGraph = adj.mapPartitionsWithIndex((index, it) =>
      Iterator.single(DGIPartition(GraphAdjPartition(index, it), $(useSecondOrder))))

    dgiGraph.persist($(storageLevel))
    dgiGraph.foreachPartition(_ => Unit)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[DGIPartition]
    SparkSession.builder().getOrCreate().createDataset(dgiGraph)
  }

  override
  def fit(model: DGIModel, psModel: DGIPSModel, graph: Dataset[_]): Unit = {
    val optim = getOptimizer

    for (curEpoch <- 1 to $(numEpoch)) {
      val (lossSum, totalTrain) = graph.rdd.map(_.asInstanceOf[DGIPartition]
        .trainEpoch(curEpoch, $(batchSize), model, psModel, $(featureDim), optim, $(numSamples)))
        .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
      println(s"curEpoch=$curEpoch train loss=${lossSum / totalTrain}")
    }

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
