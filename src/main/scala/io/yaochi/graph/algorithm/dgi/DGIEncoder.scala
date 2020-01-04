package io.yaochi.graph.algorithm.dgi

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import io.yaochi.graph.util.LayerUtil

class DGIEncoder(batchSize: Int,
                 inputDim: Int,
                 outputDim: Int,
                 weights: Array[Float],
                 start: Int = 0,
                 reshape: Boolean = false) {

  private val posConvModule = buildConvModule()
  private val negConvModule = buildConvModule()

  private val linearLayer = buildLinearLayer()
  private val preluLayer = buildPReluLayer()
  private val posLinearModule = buildLinearModule()
  private val negLinearModule = buildLinearModule()

  def forward(posX: Tensor[Float],
              negX: Tensor[Float],
              srcIndices: Tensor[Int],
              dstIndices: Tensor[Int]): Table = {
    val posConvOutput = posConvModule.forward(T.apply(Array(posX, srcIndices, dstIndices)))
    val negConvOutput = negConvModule.forward(T.apply(Array(negX, srcIndices, dstIndices)))

    T.apply(Array(
      posLinearModule.forward(T.apply(Array(posConvOutput, posX))),
      negLinearModule.forward(T.apply(Array(posConvOutput, posX)))
    ))
  }

  def backward(posX: Tensor[Float],
               negX: Tensor[Float],
               srcIndices: Tensor[Int],
               dstIndices: Tensor[Int],
               gradOutput: Table): Table = {
    val posLinearGradTable = posLinearModule.backward(T.apply(Array(posConvModule.output, posX)),
      gradOutput[Tensor[Float]](1))
    val negLinearGradTable = negLinearModule.backward(T.apply(Array(negLinearModule.output, negX)),
      gradOutput[Tensor[Float]](2))



    null
  }

  private def buildConvModule(): Sequential[Float] = {
    var module = Sequential[Float]()
    if (reshape) {
      module = module.add(Reshape(Array(batchSize, inputDim)))
    }
    module.add(new ScatterMean[Float](batchSize, inputDim))
  }

  private def buildLinearLayer(): Linear[Float] = {
    LayerUtil.buildLinear(2 * inputDim, outputDim, weights, true, start)
  }

  private def buildPReluLayer(): PReLU[Float] = {
    LayerUtil.buildPReluLayer(outputDim, weights, start + 2 * inputDim * outputDim + outputDim)
  }

  private def buildLinearModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(JoinTable(2, 2))
      .add(linearLayer)
      .add(Normalize(2.0))
      .add(preluLayer)
  }

  def getParameterSize: Int = 2 * inputDim * outputDim + 2 * outputDim
}

object DGIEncoder {
  def apply(batchSize: Int,
            inputDim: Int,
            outputDim: Int,
            weights: Array[Float],
            start: Int = 0,
            reshape: Boolean = false): DGIEncoder = new DGIEncoder(batchSize, inputDim, outputDim, weights, start, reshape)
}