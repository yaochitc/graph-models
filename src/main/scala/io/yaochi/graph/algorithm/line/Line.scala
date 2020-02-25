package io.yaochi.graph.algorithm.line

import io.yaochi.graph.algorithm.base.{Edge, GNN}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}

class Line extends GNN[LinePSModel, LineModel] {

  def makeModel(): LineModel = ???

  def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: LineModel): LinePSModel = ???

  def makeGraph(edges: RDD[Edge], model: LinePSModel, hasType: Boolean, hasWeight: Boolean): Dataset[_] = ???

  override def initialize(edgeDF: DataFrame, featureDF: DataFrame): (LineModel, LinePSModel, Dataset[_]) = ???

  override def fit(model: LineModel, psModel: LinePSModel, graph: Dataset[_]): Unit = {

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
