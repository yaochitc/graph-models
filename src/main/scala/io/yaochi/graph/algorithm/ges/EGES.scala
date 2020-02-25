package io.yaochi.graph.algorithm.ges

import io.yaochi.graph.algorithm.base.{Edge, GNN}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}

class EGES extends GNN[EGESPSModel, EGESModel] {

  def makeModel(): EGESModel = ???

  def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: EGESModel): EGESPSModel = ???

  def makeGraph(edges: RDD[Edge], model: EGESPSModel, hasType: Boolean, hasWeight: Boolean): Dataset[_] = ???

  override def initialize(edgeDF: DataFrame, featureDF: DataFrame): (EGESModel, EGESPSModel, Dataset[_]) = ???

  override def fit(model: EGESModel, psModel: EGESPSModel, graph: Dataset[_]): Unit = ???

}
