package io.yaochi.graph.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasInputDim extends Params {

  final val inputDim = new IntParam(this, "inputDim", "inputDim")

  final def getInputDim: Int = $(inputDim)

  setDefault(inputDim, 0)

  final def setInputDim(dim: Int): this.type = set(inputDim, dim)

}
