package io.yaochi.graph.algorithm.dgi

import com.intel.analytics.bigdl.tensor.Tensor

class DGIEncoder(inputDim: Int,
                 outputDim: Int,
                 weights: Array[Float],
                 start: Int = 0,
                 reshape: Boolean = false) {

  def forward(posX: Tensor[Float],
              negX: Tensor[Float],
              srcIndices: Tensor[Int],
              dstIndices: Tensor[Int]): Tensor[Float] = {
    null
  }

  def backward(x: Tensor[Float],
               srcIndices: Tensor[Int],
               dstIndices: Tensor[Int],
               gradOutput: Tensor[Float]): Tensor[Float] = {
    null
  }

  def getParameterSize: Int = 2 * inputDim * outputDim + outputDim
}

object DGIEncoder {
  def apply(inputDim: Int,
            outputDim: Int,
            weights: Array[Float],
            start: Int = 0,
            reshape: Boolean = false): DGIEncoder = new DGIEncoder(inputDim, outputDim, weights, start, reshape)
}