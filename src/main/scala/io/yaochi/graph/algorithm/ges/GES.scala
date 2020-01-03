package io.yaochi.graph.algorithm.ges

import io.yaochi.graph.algorithm.base.GNN
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class GES extends GNN[GESPSModel, GESModel] {
  override def fit(model: GESModel, psModel: GESPSModel, graph: Dataset[_]): Unit = ???

  override def makeModel(): GESModel = ???

  override def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: GESModel): GESPSModel = ???

  override def makeGraph(edges: RDD[(Long, Long)], model: GESPSModel): Dataset[_] = ???
}
