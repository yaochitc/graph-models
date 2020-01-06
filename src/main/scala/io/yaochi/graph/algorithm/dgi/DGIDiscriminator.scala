package io.yaochi.graph.algorithm.dgi

import com.intel.analytics.bigdl.nn.{MM, Sequential, Sigmoid}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

class DGIDiscriminator {

  private val mmLayer = MM[Float]()
  private val posMultiplyModule = buildMultiplyModule()
  private val negMultiplyModule = buildMultiplyModule()

  def forward(posZ: Tensor[Float],
              negZ: Tensor[Float],
              summary: Tensor[Float],
              weight: Tensor[Float]): Table = {

    val weightedSummaryTensor = mmLayer.forward(T.apply(Array(weight, summary)))

    T.apply(Array(
      posMultiplyModule.forward(T.apply(Array(posZ, weightedSummaryTensor))),
      negMultiplyModule.forward(T.apply(Array(negZ, weightedSummaryTensor)))
    ))
  }

  def backward(posZ: Tensor[Float],
               negZ: Tensor[Float],
               summary: Tensor[Float],
               weight: Tensor[Float]): Table = {
    null
  }

  private def buildMultiplyModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(MM())
      .add(Sigmoid())
  }
}
