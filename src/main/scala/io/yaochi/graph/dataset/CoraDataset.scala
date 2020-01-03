package io.yaochi.graph.dataset

import java.nio.file.Paths

import io.yaochi.graph.util.GraphIO
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.immutable.HashMap

object CoraDataset {
  def load(directory: String): (DataFrame, DataFrame, DataFrame) = {
    val ss = SparkSession.builder().getOrCreate()

    val contentFilename = "cora.content"
    val contentInput = Paths.get(directory, contentFilename)

    val nodeSchema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("feature", StringType, nullable = false),
      StructField("label", FloatType, nullable = false)
    ))

    val nodeRDD = ss.sparkContext.textFile(contentInput.toString)
      .map(line => line.split("\t"))

    val nodes = nodeRDD.map(fields => fields.head).distinct()
      .collect()
      .sorted

    val labels = nodeRDD.map(fields => fields.last).distinct()
      .collect()
      .sorted

    val node2Id = HashMap(nodes.view.zipWithIndex: _*)
    val node2IdBc = ss.sparkContext.broadcast(node2Id)

    val label2Id = HashMap(labels.view.zipWithIndex: _*)
    val label2IdBc = ss.sparkContext.broadcast(label2Id)

    val encodedNodeRDD = nodeRDD.map(fields => {
      val len = fields.length
      val id = node2IdBc.value(fields.head).toLong
      val features = fields.slice(1, len - 1).map(_.toFloat)
      val featureSum = features.sum + 1e-15
      val normFeatures = features.map(feature => feature / featureSum)
        .mkString(" ")
      val label = label2IdBc.value(fields.last).toFloat
      Row(id, normFeatures, label)
    })

    val citesFilename = "cora.cites"
    val citesInput = Paths.get(directory, citesFilename)

    val edgeSchema = StructType(Seq(
      StructField("src", LongType, nullable = false),
      StructField("dst", LongType, nullable = false)
    ))

    val edgeRDD = ss.sparkContext.textFile(citesInput.toString)
      .map(line => line.split("\t"))
      .map(fields => Row(
        node2IdBc.value(fields(0)).toLong,
        node2IdBc.value(fields(1)).toLong
      ))

    val featureLabelDF = ss.createDataFrame(encodedNodeRDD, nodeSchema)

    (ss.createDataFrame(edgeRDD, edgeSchema),
      featureLabelDF.select("node", "feature"),
      featureLabelDF.select("node", "label")
    )
  }

  private def loadEdges(directory: String): DataFrame = {
    val filename = "cora.cites"
    val input = Paths.get(directory, filename)

    GraphIO.loadEdges(input.toString, isWeighted = false, isFeatured = false, isTyped = false, sep = "\t")
  }

  private def loadNodes(directory: String): DataFrame = {
    val filename = "cora.content"
    val input = Paths.get(directory, filename)

    GraphIO.loadNodes(input.toString, isWeighted = false, isFeatured = true, isTyped = false, sep = ",")
  }
}
