package io.yaochi.graph.dataset

import java.nio.file.Paths

import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.immutable.HashMap

object ArxivDataset {
  def load(directory: String): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()

    val filename = "ca-AstroPh.txt"
    val input = Paths.get(directory, filename)

    val schema = StructType(Seq(
      StructField("src", LongType, nullable = false),
      StructField("dst", LongType, nullable = false)
    ))

    val rdd = ss.sparkContext.textFile(input.toString)
      .zipWithIndex()
      .filter(x => x._2 > 3)
      .map(line => line._1.split("\t").map(f => f.toLong))

    val nodes = rdd.flatMap(fields => fields.slice(0, 2)).distinct()
      .collect()
      .sorted

    val node2Id = HashMap(nodes.view.zipWithIndex: _*)
    val node2IdBc = ss.sparkContext.broadcast(node2Id)

    val encodedRDD = rdd.map(fields => {
      Row(
        node2IdBc.value(fields(0)).toLong,
        node2IdBc.value(fields(1)).toLong
      )
    })

    ss.createDataFrame(encodedRDD, schema)
  }

}
