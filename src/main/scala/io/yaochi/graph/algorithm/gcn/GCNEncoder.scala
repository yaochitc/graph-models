package io.yaochi.graph.algorithm.gcn

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.graph.util.LayerUtil

class GCNEncoder(batchSize: Int,
                 inputDim: Int,
                 outputDim: Int,
                 weights: Array[Float],
                 start: Int = 0,
                 reshape: Boolean = false) {
  private val linearLayer = LayerUtil.buildLinear(inputDim, outputDim, weights, false, start)
  private val linearModule = buildLinearModule()

  private val convLayer = new Scatter[Float](batchSize, outputDim)
  private val biasLayer = LayerUtil.buildBiasLayer(outputDim, weights, start + inputDim * outputDim)
  private val convModule = buildConvModule()

  def forward(x: Tensor[Float],
              srcIndices: Tensor[Int],
              dstIndices: Tensor[Int],
              norms: Tensor[Float]): Tensor[Float] = {
    val linearOutput = linearModule.forward(x)
      .toTensor[Float]
    convModule.forward(T.array(Array(linearOutput, norms, srcIndices, dstIndices)))
      .toTensor[Float]
  }

  def backward(x: Tensor[Float],
               srcIndices: Tensor[Int],
               dstIndices: Tensor[Int],
               norms: Tensor[Float],
               gradOutput: Tensor[Float]): Tensor[Float] = {
    val linearOutput = linearModule.output.toTensor[Float]
    val gradTable = convModule.backward(T.array(Array(linearOutput, norms, srcIndices, dstIndices)), gradOutput).toTable

    val gradTensor = linearModule.backward(x, gradTable[Tensor[Float]](1)).toTensor[Float]

    val gradWeight = linearLayer.gradWeight
    val gradBias = biasLayer.gradBias

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

    gradTensor
  }

  private def buildLinearModule(): Sequential[Float] = {
    var module = Sequential[Float]()
    if (reshape) {
      module = module.add(Reshape(Array(batchSize, inputDim)))
    }
    module.add(linearLayer)
  }

  private def buildConvModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(convLayer)
      .add(biasLayer)
  }

  def getParameterSize: Int = inputDim * outputDim + outputDim
}

object GCNEncoder {
  def apply(batchSize: Int,
            inputDim: Int,
            outputDim: Int,
            weights: Array[Float],
            start: Int = 0,
            reshape: Boolean = false): GCNEncoder = new GCNEncoder(batchSize, inputDim, outputDim, weights, start, reshape)
}
