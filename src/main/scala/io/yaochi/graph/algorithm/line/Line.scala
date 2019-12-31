package io.yaochi.graph.algorithm.line

import io.yaochi.graph.algorithm.base.GNN
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class Line extends GNN[LinePSModel] {

  override def makeModel(minId: Long, maxId: Long, index: RDD[Long]): LinePSModel = ???

  override def makeGraph(edges: RDD[(Long, Long)], model: LinePSModel): Dataset[_] = ???

  override def fit(model: LinePSModel, graph: Dataset[_]): Unit = {

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}
