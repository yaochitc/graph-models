package io.yaochi.graph.algorithm.ges

import io.yaochi.graph.algorithm.base.{Edge, GNN}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class GES extends GNN[GESPSModel, GESModel] {
  override def fit(model: GESModel, psModel: GESPSModel, graph: Dataset[_]): Unit = ???

  override def makeModel(): GESModel = ???

  override def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: GESModel): GESPSModel = ???

  override def makeGraph(edges: RDD[Edge], model: GESPSModel, hasWeight: Boolean, hasType: Boolean): Dataset[_] = ???
}
