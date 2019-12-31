package io.yaochi.graph.dataset

import java.nio.file.Paths

import io.yaochi.graph.util.GraphIO
import org.apache.spark.sql.DataFrame

object BlogCatalogDataset {
  def load(directory: String): (DataFrame, DataFrame) = {
    val edges = loadEdges(directory)
    val nodes = loadNodes(directory)

    (edges, nodes)
  }

  private def loadEdges(directory: String): DataFrame = {
    val filename = "edges.csv"
    val input = Paths.get(directory, filename)

    GraphIO.loadEdges(input.toString, isWeighted = false, isFeatured = false, isTyped = false, sep = ",")
  }

  private def loadNodes(directory: String): DataFrame = {
    val filename = "group-edges.csv"
    val input = Paths.get(directory, filename)

    GraphIO.loadNodes(input.toString, isWeighted = false, isFeatured = true, isTyped = false, sep = ",")
  }
}
