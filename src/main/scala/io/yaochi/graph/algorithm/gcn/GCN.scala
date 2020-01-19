package io.yaochi.graph.algorithm.gcn

import io.yaochi.graph.algorithm.base.{Edge, GraphAdjPartition, SupervisedGNN}
import io.yaochi.graph.params.{HasHiddenDim, HasNumClasses, HasTestRatio}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

class GCN extends SupervisedGNN[GCNPSModel, GCNModel]
  with HasHiddenDim with HasNumClasses
  with HasTestRatio {

  override def makeModel(): GCNModel = GCNModel($(featureDim), $(hiddenDim), $(numClasses))

  override def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: GCNModel): GCNPSModel = {
    GCNPSModel.apply(minId, maxId, model.getParameterSize, getOptimizer,
      index, $(psPartitionNum), $(useBalancePartition))
  }

  override def makeGraph(edges: RDD[Edge], model: GCNPSModel, hasType: Boolean, hasWeight: Boolean): Dataset[_] = {
    // build adj graph partitions
    val adjGraph = edges.map(f => (f.src, f)).groupByKey($(partitionNum))
      .mapPartitionsWithIndex((index, it) =>
        Iterator.single(GraphAdjPartition.apply(index, it, hasType, hasWeight)))

    adjGraph.persist($(storageLevel))
    adjGraph.foreachPartition(_ => Unit)
    adjGraph.map(_.init(model, $(numBatchInit))).reduce(_ + _)

    // build GCN graph partitions
    val gcnGraph = adjGraph.map(GCNPartition(_, model, $(testRatio)))
    gcnGraph.persist($(storageLevel))
    gcnGraph.count()
    adjGraph.unpersist(true)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[GCNPartition]
    SparkSession.builder().getOrCreate().createDataset(gcnGraph)
  }

  override def fit(model: GCNModel, psModel: GCNPSModel, graph: Dataset[_]): Unit = {
    val optim = getOptimizer

    val (trainSize, testSize) = graph.rdd.map(_.asInstanceOf[GCNPartition].getTrainTestSize)
      .reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
    println(s"numTrain=$trainSize numTest=$testSize testRatio=${$(testRatio)} samples=${$(numSamples)}")

    for (curEpoch <- 1 to $(numEpoch)) {
      val (lossSum, trainRight) = graph.rdd.map(_.asInstanceOf[GCNPartition].trainEpoch(curEpoch, $(batchSize), model, psModel,
        $(featureDim), optim, $(numSamples))).reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
      val predRight = graph.rdd.map(_.asInstanceOf[GCNPartition].predictEpoch(curEpoch, $(batchSize) * 10, model, psModel,
        $(featureDim), $(numSamples))).reduce(_ + _)
      println(s"curEpoch=$curEpoch " +
        s"train loss=${lossSum / trainSize} " +
        s"train acc=${trainRight.toDouble / trainSize} " +
        s"test acc=${predRight.toDouble / testSize}")
    }
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)


}
