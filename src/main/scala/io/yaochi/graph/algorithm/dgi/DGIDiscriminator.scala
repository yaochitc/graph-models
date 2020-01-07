package io.yaochi.graph.algorithm.dgi

import com.intel.analytics.bigdl.nn.{MM, Mean, Sequential, Sigmoid}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import io.yaochi.graph.util.{BackwardUtil, LayerUtil}

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

    T.apply(
      posMultiplyModule.forward(T.apply(posZ, summary)),
      negMultiplyModule.forward(T.apply(negZ, summary))
    )
  }

  def backward(posZ: Tensor[Float],
               negZ: Tensor[Float],
               gradOutput: Table): Table = {
    val summary = summaryModule.output.toTensor[Float]
    val posMultiplyGradTable = posMultiplyModule.backward(T.apply(posZ, summary),
      gradOutput[Tensor[Float]](1)
    ).toTable
    val negMultiplyGradTable = negMultiplyModule.backward(T.apply(negZ, summary),
      gradOutput[Tensor[Float]](2)
    ).toTable

    val negGradTensor = negMultiplyGradTable[Tensor[Float]](1)
    val posGradTensor = posMultiplyGradTable[Tensor[Float]](1).add(
      summaryModule.backward(posZ, negMultiplyGradTable[Tensor[Float]](2)).toTensor[Float]
    ).add(
      summaryModule.backward(posZ, posMultiplyGradTable[Tensor[Float]](2)).toTensor[Float]
    )

    BackwardUtil.linearBackward(linearLayer, weights, start)

    T.apply(posGradTensor, negGradTensor)
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

  def getParameterSize: Int = inputDim * inputDim
}

object DGIDiscriminator {
  def apply(inputDim: Int,
            weights: Array[Float],
            start: Int): DGIDiscriminator = new DGIDiscriminator(inputDim, weights, start)
}