package io.yaochi.graph.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasHiddenDim extends Params {

  final val hiddenDim = new IntParam(this, "hiddenDim", "hiddenDim")

  final def getHiddenDim: Int = $(hiddenDim)

  setDefault(hiddenDim, 0)

  final def setHiddenDim(dim: Int): this.type = set(hiddenDim, dim)

}
