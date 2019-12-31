package io.yaochi.graph.algorithm.line

import com.tencent.angel.spark.models.PSMatrix
import io.yaochi.graph.algorithm.base.GNNPSModel

class LinePSModel(graph: PSMatrix,
                  val embedding: PSMatrix) extends GNNPSModel(graph) {

}
