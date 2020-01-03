package io.yaochi.graph.algorithm.ges

import io.yaochi.graph.algorithm.base.GNN
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class EGES extends GNN[EGESPSModel, EGESModel] {
  override def fit(model: EGESModel, psModel: EGESPSModel, graph: Dataset[_]): Unit = ???

  override def makeModel(): EGESModel = ???

  override def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: EGESModel): EGESPSModel = ???

  override def makeGraph(edges: RDD[(Long, Long)], model: EGESPSModel): Dataset[_] = ???

}
