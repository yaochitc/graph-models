package io.yaochi.graph.algorithm.ges

import io.yaochi.graph.algorithm.base.{Edge, GNN}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}

class GES extends GNN[GESPSModel, GESModel] {

  def makeModel(): GESModel = ???

  def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: GESModel): GESPSModel = ???

  def makeGraph(edges: RDD[Edge], model: GESPSModel, hasType: Boolean, hasWeight: Boolean): Dataset[_] = ???

  override def initialize(edgeDF: DataFrame, featureDF: DataFrame): (GESModel, GESPSModel, Dataset[_]) = ???

  override def fit(model: GESModel, psModel: GESPSModel, graph: Dataset[_]): Unit = ???

}
