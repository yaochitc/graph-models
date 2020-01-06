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
              dstIndices: Tensor[Int],
              counts: Tensor[Float]): Table = {
    val (posInput, negInput) = if (reshape) {
      (posReshapeLayer.forward(posX), negReshapeLayer.forward(negX))
    } else {
      (posX, negX)
    }
    val posConvOutput = posConvLayer.forward(T.apply(posInput, srcIndices, dstIndices, counts))
    val negConvOutput = negConvLayer.forward(T.apply(negInput, srcIndices, dstIndices, counts))

    T.apply(
      posLinearModule.forward(T.apply(posConvOutput, posInput)),
      negLinearModule.forward(T.apply(negConvOutput, negInput))
    )
  }

  def backward(posX: Tensor[Float],
               negX: Tensor[Float],
               srcIndices: Tensor[Int],
               dstIndices: Tensor[Int],
               counts: Tensor[Float]
               gradOutput: Table): Table = {
    val (posInput, negInput) = if (reshape) {
      (posReshapeLayer.output, negReshapeLayer.output)
    } else {
      (posX, negX)
    }

    val posLinearGradTable = posLinearModule.backward(T.apply(posConvLayer.output, posInput),
      gradOutput[Tensor[Float]](1)).toTable
    val negLinearGradTable = negLinearModule.backward(T.apply(negConvLayer.output, negInput),
      gradOutput[Tensor[Float]](2)).toTable

    val posConvGradTable = posConvLayer.backward(T.apply(posInput, srcIndices, dstIndices, counts),
      posLinearGradTable[Tensor[Float]](1)).toTable
    val negConvGradTable = negConvLayer.backward(T.apply(negInput, srcIndices, dstIndices, counts),
      negLinearGradTable[Tensor[Float]](1)).toTable

    val posInputGrad = posLinearGradTable[Tensor[Float]](2).add(
      posConvGradTable[Tensor[Float]](1)
    )
    val negInputGrad = negLinearGradTable[Tensor[Float]](2).add(
      negConvGradTable[Tensor[Float]](1)
    )

    val gradWeight = linearLayer.gradWeight
    val gradBias = linearLayer.gradBias

    val gradWeightSize = gradWeight.size()
    val outputSize = gradWeightSize(0)
    val inputSize = gradWeightSize(1)

    var curOffset = start
    for (i <- 0 until outputSize; j <- 0 until inputSize) {
      weights(curOffset + i * inputSize + j) = gradWeight.valueAt(i + 1, j + 1)
    }
    curOffset += outputSize * inputSize

    for (i <- 0 until outputSize) {
      weights(curOffset + i) = gradBias.valueAt(i + 1)
    }
    curOffset += outputSize

    val preluGradWeight = preluLayer.gradWeight
    for (i <- 0 until outputSize) {
      weights(curOffset + i) = preluGradWeight.valueAt(i + 1)
    }

    if (reshape) {
      T.apply(
        posReshapeLayer.backward(posX, posInputGrad),
        negReshapeLayer.backward(negX, negInputGrad)
      )
    } else {
      T.apply(posInputGrad, negInputGrad)
    }
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