package io.yaochi.graph.algorithm.base

import com.tencent.angel.spark.ml.graph.params._
import io.yaochi.graph.data.SampleParser
import io.yaochi.graph.params._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}

abstract class SupervisedGNN[PSModel <: SupervisedGNNPSModel, Model <: GNNModel](val uid: String) extends Serializable
  with HasBatchSize with HasFeatureDim with HasOptimizer
  with HasNumEpoch with HasNumSamples with HasNumBatchInit
  with HasPartitionNum with HasPSPartitionNum with HasUseBalancePartition
  with HasDataFormat with HasStorageLevel {

  def this() = this(Identifiable.randomUID("SupervisedGNN"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  def initFeatures(model: PSModel, features: Dataset[Row], minId: Long, maxId: Long): Unit = {
    features.select("id", "feature")
      .rdd.map(row => (row.getLong(0), row.getString(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .map(f => (f._1, SampleParser.parseFeature(f._2, $(featureDim), $(dataFormat))))
      .mapPartitions(it =>
        Iterator(NodeFeaturePartition.apply(it)))
      .map(_.init(model, $(numBatchInit))).count()
  }

  def initLabels(model: PSModel, labels: Dataset[Row], minId: Long, maxId: Long): Unit = {
    labels.rdd.map(row => (row.getLong(0), row.getFloat(1)))
      .filter(f => f._1 >= minId && f._1 <= maxId)
      .mapPartitions(it =>
        Iterator(NodeLabelPartition.apply(it, model.dim)))
      .map(_.init(model)).count()
  }

  def makeEdges(edgeDF: DataFrame): RDD[(Long, Long)] = {
    edgeDF.select("src", "dst").rdd
      .map(row => (row.getLong(0), row.getLong(1)))
      .filter(f => f._1 != f._2)
  }

  def initialize(edgeDF: DataFrame,
                 featureDF: DataFrame,
                 labelDF: Option[DataFrame]): (Model, PSModel, Dataset[_])

  def fit(model: Model, psModel: PSModel, graph: Dataset[_]): Unit
}
