package io.yaochi.graph.algorithm.line

import io.yaochi.graph.algorithm.base.GNN
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class Line extends GNN[LinePSModel, LineModel] {

  override def makeModel(): LineModel = ???

  override def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: LineModel): LinePSModel = ???

  override def makeGraph(edges: RDD[(Long, Long)], model: LinePSModel): Dataset[_] = ???

  override def fit(model: LineModel, psModel: LinePSModel, graph: Dataset[_]): Unit = {

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
