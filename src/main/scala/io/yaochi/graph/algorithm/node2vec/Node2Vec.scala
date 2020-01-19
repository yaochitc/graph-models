package io.yaochi.graph.algorithm.node2vec

import io.yaochi.graph.algorithm.base.{Edge, GNN}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class Node2Vec extends GNN[Node2VecPSModel, Node2VecModel] {

  override def makeModel(): Node2VecModel = ???

  override def makePSModel(minId: Long, maxId: Long, index: RDD[Long], model: Node2VecModel): Node2VecPSModel = ???

  override def makeGraph(edges: RDD[Edge], model: Node2VecPSModel, hasWeight: Boolean, hasType: Boolean): Dataset[_] = ???

  override def fit(model: Node2VecModel, PSModel: Node2VecPSModel, graph: Dataset[_]): Unit = {

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

}
