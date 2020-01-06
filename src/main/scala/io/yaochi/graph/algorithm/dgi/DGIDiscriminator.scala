package io.yaochi.graph.algorithm.dgi

import com.intel.analytics.bigdl.nn.{MM, Mean, Sequential, Sigmoid}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import io.yaochi.graph.util.LayerUtil

class DGIDiscriminator(inputDim: Int,
                       weights: Array[Float],
                       start: Int) {

  private val linearLayer = LayerUtil.buildLinear(inputDim, inputDim, weights, false, start)
  private val summaryModule = buildSummaryModule()
  private val posMultiplyModule = buildMultiplyModule()
  private val negMultiplyModule = buildMultiplyModule()

  def forward(posZ: Tensor[Float],
              negZ: Tensor[Float]): Table = {
    val summary = summaryModule.forward(posZ)

    T.apply(Array(
      posMultiplyModule.forward(T.apply(posZ, summary)),
      negMultiplyModule.forward(T.apply(negZ, summary))
    ))
  }

  def backward(posZ: Tensor[Float],
               negZ: Tensor[Float]): Table = {
    null
  }

  private def buildSummaryModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Mean(squeeze = false))
      .add(linearLayer)
      .add(Sigmoid())
  }

  private def buildMultiplyModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(MM(transB = true))
      .add(Sigmoid())
  }
}

object DGIDiscriminator {
  def apply(inputDim: Int,
            weights: Array[Float],
            start: Int): DGIDiscriminator = new DGIDiscriminator(inputDim, weights, start)
}