package io.yaochi.graph.dataset

import java.nio.file.Paths

import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

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
      .map(line => line._1.split("\t"))
      .map(fields => Row(fields(0).toLong, fields(1).toLong))
    ss.createDataFrame(rdd, schema)
  }

}
