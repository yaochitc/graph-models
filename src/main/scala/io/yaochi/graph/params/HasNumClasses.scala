package io.yaochi.graph.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasNumClasses extends Params {

  final val numClasses = new IntParam(this, "numClasses", "numClasses")

  final def getNumClasses: Int = $(numClasses)

  setDefault(numClasses, 0)

  final def setNumClasses(dim: Int): this.type = set(numClasses, dim)

}
