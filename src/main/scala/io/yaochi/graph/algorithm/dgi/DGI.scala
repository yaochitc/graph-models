package io.yaochi.graph.algorithm.dgi

import com.tencent.angel.spark.context.PSContext
import io.yaochi.graph.algorithm.base.{GNN, GraphAdjPartition}
import io.yaochi.graph.params.{HasHiddenDim, HasOutputDim, HasUseSecondOrder}
import io.yaochi.graph.util.DataLoaderUtils
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.storage.StorageLevel

class DGI extends GNN[DGIPSModel, DGIModel]
  with HasHiddenDim
  with HasOutputDim with HasUseSecondOrder {

  def makeModel(): DGIModel = DGIModel($(featureDim), $(hiddenDim), $(outputDim))

  def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: DGIModel): DGIPSModel = {
    DGIPSModel.apply(minId, maxId, model.getParameterSize, getOptimizer,
      index, $(psPartitionNum), $(useBalancePartition))
  }

  def makeGraph(edges: RDD[(Long, Long)], model: DGIPSModel): Dataset[_] = {
    val adj = edges.map(f => (f._1, f)).groupByKey($(partitionNum))

    if ($(useSecondOrder)) {
      // if second order is required, init neighbors on PS
      adj.mapPartitions(it =>
        Iterator(GraphAdjPartition(it)))
        .map(_.init(model, $(numBatchInit))).reduce(_ + _)
    }

    val dgiGraph = adj.mapPartitions(it =>
      Iterator.single(DGIPartition(GraphAdjPartition(it), $(useSecondOrder))))

    dgiGraph.persist($(storageLevel))
    dgiGraph.foreachPartition(_ => Unit)

    implicit val encoder = org.apache.spark.sql.Encoders.kryo[DGIPartition]
    SparkSession.builder().getOrCreate().createDataset(dgiGraph)
  }

  override def initialize(edgeDF: DataFrame, featureDF: DataFrame): (DGIModel, DGIPSModel, Dataset[_]) = {
    val start = System.currentTimeMillis()

    // read edges
    val edges = makeEdges(edgeDF)

    edges.persist(StorageLevel.DISK_ONLY)

    val (minId, maxId, numEdges) = edges.mapPartitions(DataLoaderUtils.summarizeApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)
    val index = edges.flatMap(f => Iterator(f._1, f._2))
    println(s"minId=$minId maxId=$maxId numEdges=$numEdges")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    val model = makeModel()

    val psModel = makePSModel(minId, maxId + 1, index, model)
    psModel.initialize()

    initFeatures(psModel, featureDF, minId, maxId)

    val graph = makeGraph(edges, psModel)

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")

    (model, psModel, graph)
  }

  override def fit(model: DGIModel, psModel: DGIPSModel, graph: Dataset[_]): Unit = {
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
