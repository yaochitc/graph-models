package io.yaochi.graph.algorithm.base

import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.graph.params._
import io.yaochi.graph.data.SampleParser
import io.yaochi.graph.params._
import io.yaochi.graph.util.DataLoaderUtils
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

abstract class GNN[PSModel <: GNNPSModel](val uid: String) extends Serializable
  with HasBatchSize with HasFeatureDim with HasOptimizer
  with HasNumEpoch with HasNumSamples with HasNumBatchInit
  with HasPartitionNum with HasPSPartitionNum with HasUseBalancePartition
  with HasDataFormat with HasStorageLevel {

  def this() = this(Identifiable.randomUID("GNN"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  def fit(model: PSModel, graph: Dataset[_]): Unit

  def initFeatures(model: PSModel, features: Dataset[Row], minId: Long, maxId: Long): Unit = {
    features.rdd.filter(row => row.length > 0)
      .filter(row => row.get(0) != null)
      .map(row => (row.getLong(0), row.getString(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .map(f => (f._1, SampleParser.parseFeature(f._2, $(featureDim), $(dataFormat))))
      .mapPartitionsWithIndex((index, it) =>
        Iterator(NodeFeaturePartition.apply(index, it)))
      .map(_.init(model, $(numBatchInit))).count()
  }

  def initLabels(model: PSModel, labels: Dataset[Row], minId: Long, maxId: Long): Unit = {
    labels.rdd.map(row => (row.getLong(0), row.getFloat(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .mapPartitionsWithIndex((index, it) =>
        Iterator(NodeLabelPartition.apply(index, it, model.dim)))
      .map(_.init(model)).count()
  }

  def makeModel(minId: Long, maxId: Long, index: RDD[Long]): PSModel

  def makeGraph(edges: RDD[(Long, Long)], model: PSModel): Dataset[_]

  def initialize(edgeDF: DataFrame, featureDF: DataFrame): (PSModel, Dataset[_]) =
    initialize(edgeDF, featureDF, None)

  def initialize(edgeDF: DataFrame,
                 featureDF: DataFrame,
                 labelDF: Option[DataFrame]): (PSModel, Dataset[_]) = {
    val start = System.currentTimeMillis()

    // read edges
    val edges = edgeDF.select("src", "dst").rdd
      .map(row => (row.getLong(0), row.getLong(1)))
      .filter(f => f._1 != f._2)

    edges.persist(StorageLevel.DISK_ONLY)

    val (minId, maxId, numEdges) = edges.mapPartitions(DataLoaderUtils.summarizeApplyOp)
      .reduce(DataLoaderUtils.summarizeReduceOp)
    val index = edges.flatMap(f => Iterator(f._1, f._2))
    println(s"minId=$minId maxId=$maxId numEdges=$numEdges")

    PSContext.getOrCreate(SparkContext.getOrCreate())

    val model = makeModel(minId, maxId + 1, index)

    labelDF.foreach(f => initLabels(model, f, minId, maxId))
    initFeatures(model, featureDF, minId, maxId)

    val graph = makeGraph(edges, model)

    val end = System.currentTimeMillis()
    println(s"initialize cost ${(end - start) / 1000}s")

    (model, graph)
  }
}
