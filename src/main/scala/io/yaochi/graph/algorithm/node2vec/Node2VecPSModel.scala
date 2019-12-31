package io.yaochi.graph.algorithm.node2vec

import com.tencent.angel.spark.models.PSMatrix
import io.yaochi.graph.algorithm.base.GNNPSModel

class Node2VecPSModel (graph: PSMatrix,
                       embedding: PSMatrix) extends GNNPSModel(graph) {

}
