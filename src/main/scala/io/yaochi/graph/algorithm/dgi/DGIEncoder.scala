package io.yaochi.graph.algorithm.dgi

import com.intel.analytics.bigdl.nn.{ScatterMean, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import io.yaochi.graph.util.LayerUtil

class DGIEncoder(batchSize: Int,
                 inputDim: Int,
                 outputDim: Int,
                 weights: Array[Float],
                 start: Int = 0,
                 reshape: Boolean = false) {

  private var posReshapeLayer: Reshape[Float] = buildReshapeLayer()
  private var negReshapeLayer: Reshape[Float] = buildReshapeLayer()

  private val posConvLayer = ScatterMean[Float](batchSize, inputDim)
  private val negConvLayer = ScatterMean[Float](batchSize, inputDim)

  private val linearLayer = LayerUtil.buildLinear(2 * inputDim, outputDim, weights, true, start)
  private val preluLayer = LayerUtil.buildPReluLayer(outputDim, weights, start + 2 * inputDim * outputDim + outputDim)
  private val posLinearModule = buildLinearModule()
  private val negLinearModule = buildLinearModule()

  def forward(posX: Tensor[Float],
              negX: Tensor[Float],
              srcIndices: Tensor[Int],
              dstIndices: Tensor[Int]): Table = {
    val (posInput, negInput) = if (reshape) {
      (posReshapeLayer.forward(posX), negReshapeLayer.forward(negX))
    } else {
      (posX, negX)
    }
    val posConvOutput = posConvLayer.forward(T.apply(posInput, srcIndices, dstIndices))
    val negConvOutput = negConvLayer.forward(T.apply(negInput, srcIndices, dstIndices))

    T.apply(
      posLinearModule.forward(T.apply(posConvOutput, posInput)),
      negLinearModule.forward(T.apply(negConvOutput, negInput))
    )
  }

  def backward(posX: Tensor[Float],
               negX: Tensor[Float],
               srcIndices: Tensor[Int],
               dstIndices: Tensor[Int],
               gradOutput: Table): Table = {
    val posLinearGradTable = posLinearModule.backward(T.apply(Array(posConvLayer.output, posX)),
      gradOutput[Tensor[Float]](1))
    val negLinearGradTable = negLinearModule.backward(T.apply(Array(negConvLayer.output, negX)),
      gradOutput[Tensor[Float]](2))


    null
  }

  private def buildReshapeLayer(): Reshape[Float] = {
    if (reshape) {
      Reshape(Array(batchSize, inputDim))
    } else {
      null
    }
  }

  private def buildLinearModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(JoinTable(2, 2))
      .add(LinearWrapper(linearLayer))
      .add(Normalize(2.0))
      .add(PReluWrapper(preluLayer))
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