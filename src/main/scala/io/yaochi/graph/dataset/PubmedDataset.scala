package io.yaochi.graph.dataset

import java.nio.file.Paths

import io.yaochi.graph.util.GraphIO
import org.apache.spark.sql.DataFrame

object PubmedDataset {
  def load(directory: String): (DataFrame, DataFrame, DataFrame) = {
    val edges = GraphIO.loadEdges(Paths.get(directory, "edges.csv").toString, isWeighted = false, isFeatured = false, isTyped = false, ",")
    val nodes = GraphIO.loadNodes(Paths.get(directory, "nodes.csv").toString, isWeighted = false, isFeatured = true, isTyped = false, ",")
    val labels = GraphIO.loadLabels(Paths.get(directory, "labels.csv").toString)

    (edges, nodes, labels)
  }
}
