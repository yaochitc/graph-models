package io.yaochi.graph.params

import org.apache.spark.ml.param.{IntParam, Params}

trait HasOutputDim extends Params {

  final val outputDim = new IntParam(this, "outputDim", "outputDim")

  final def getOutputDim: Int = $(outputDim)

  setDefault(outputDim, 0)

  final def setOutputDim(dim: Int): this.type = set(outputDim, dim)

}
