package io.yaochi.graph.util

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

object GraphIO {

  private val SEP = "sep"
  private val HEADER = "header"

  def loadEdges(input: String, isWeighted: Boolean,
                isFeatured: Boolean, isTyped: Boolean,
                sep: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()

    var schema = StructType(Seq(
      StructField("src", LongType, nullable = false),
      StructField("dst", LongType, nullable = false)
    ))

    if (isWeighted) {
      schema = schema.add(StructField("weight", FloatType, nullable = false))
    }

    if (isFeatured) {
      schema = schema.add(StructField("feature", StringType, nullable = false))
    }

    if (isTyped) {
      schema = schema.add(StructField("type", IntegerType, nullable = false))
    }

    ss.read
      .option(SEP, sep)
      .option(HEADER, "false")
      .schema(schema)
      .csv(input)
  }

  def loadNodes(input: String, isWeighted: Boolean,
                isFeatured: Boolean, isTyped: Boolean,
                sep: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()

    var schema = StructType(Seq(
      StructField("node", LongType, nullable = false)
    ))

    if (isWeighted) {
      schema = schema.add(StructField("weight", FloatType, nullable = false))
    }

    if (isFeatured) {
      schema = schema.add(StructField("feature", StringType, nullable = false))
    }

    if (isTyped) {
      schema = schema.add(StructField("type", IntegerType, nullable = false))
    }

    ss.read
      .option(SEP, sep)
      .option(HEADER, "false")
      .schema(schema)
      .csv(input)
  }

  def loadLabels(input: String,
                sep: String = " "): DataFrame = {
    val ss = SparkSession.builder().getOrCreate()

    var schema = StructType(Seq(
      StructField("node", LongType, nullable = false),
      StructField("label", LongType, nullable = false)
    ))

    ss.read
      .option(SEP, sep)
      .option(HEADER, "false")
      .schema(schema)
      .csv(input)
  }


}
