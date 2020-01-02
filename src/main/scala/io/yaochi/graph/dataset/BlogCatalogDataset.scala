package io.yaochi.graph.dataset

import java.nio.file.Paths

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}

object BlogCatalogDataset {
  def load(directory: String): (DataFrame, DataFrame) = {
    val edges = loadEdges(directory)
    val nodes = loadNodes(directory)

    (edges, nodes)
  }

  private def loadEdges(directory: String): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()

    val filename = "edges.csv"
    val input = Paths.get(directory, filename)

    val schema = StructType(Seq(
      StructField("src", LongType, nullable = false),
      StructField("dst", LongType, nullable = false)
    ))

    val rdd = ss.sparkContext.textFile(input.toString)
      .map(line => line.split(","))
      .map(fields => Row(fields(0).toLong - 1, fields(1).toLong - 1))

    ss.createDataFrame(rdd, schema)
  }

  private def loadNodes(directory: String): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()

    val filename = "group-edges.csv"
    val input = Paths.get(directory, filename)

    val schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("feature", StringType, nullable = false)
    ))

    val rdd = ss.sparkContext.textFile(input.toString)
      .map(line => line.split(","))
      .map(fields => Row(fields(0).toLong - 1, (fields(1).toLong - 1).toString))

    ss.createDataFrame(rdd, schema)
  }
}
