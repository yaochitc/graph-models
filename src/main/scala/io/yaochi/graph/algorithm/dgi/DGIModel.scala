package io.yaochi.graph.algorithm.dgi

import com.intel.analytics.bigdl.tensor.Tensor
import io.yaochi.graph.algorithm.base.GNNModel

class DGIModel(inputDim: Int,
               hiddenDim: Int,
               outputDim: Int) extends GNNModel {

  def getParameterSize: Int = {
    val conv1ParamSize = 2 * inputDim * hiddenDim + 2 * hiddenDim
    val conv2ParamSize = 2 * hiddenDim * outputDim + 2 * outputDim
    val discriminatorParamSize = outputDim * outputDim
    conv1ParamSize + conv2ParamSize + discriminatorParamSize
  }

  def backward(batchSize: Int,
               posX: Array[Float],
               negX: Array[Float],
               featureDim: Int,
               firstEdgeIndex: Array[Long],
               secondEdgeIndex: Array[Long],
               weights: Array[Float]): Float = {
    val posXTensor = Tensor.apply(posX, Array(posX.length))
    val negXTensor = Tensor.apply(negX, Array(negX.length))

    val (firstEdgeSrcIndices, firstEdgeDstIndices) = DGIModel.calcIndices(firstEdgeIndex)
    val firstEdgeSrcIndicesTensor = Tensor.apply(firstEdgeSrcIndices, Array(firstEdgeSrcIndices.length))
    val firstEdgeDstIndicesTensor = Tensor.apply(firstEdgeDstIndices, Array(firstEdgeDstIndices.length))

    val (secondEdgeSrcIndices, secondEdgeDstIndices) = DGIModel.calcIndices(secondEdgeIndex)
    val secondEdgeSrcIndicesTensor = Tensor.apply(secondEdgeSrcIndices, Array(secondEdgeSrcIndices.length))
    val secondEdgeDstIndicesTensor = Tensor.apply(secondEdgeDstIndices, Array(secondEdgeDstIndices.length))

    val numSecondOrderNodes = posX.length / inputDim
    val secondEdgeEncoder = DGIEncoder(numSecondOrderNodes, inputDim, hiddenDim, weights, reshape = true)
    var offset = secondEdgeEncoder.getParameterSize
    val firstEdgeEncoder = DGIEncoder(batchSize, hiddenDim, outputDim, weights, offset)
    offset += firstEdgeEncoder.getParameterSize
    val discriminator = DGIDiscriminator(outputDim, weights, offset)

    val secondEdgeOutputTable = secondEdgeEncoder.forward(posXTensor,
      negXTensor,
      secondEdgeSrcIndicesTensor,
      secondEdgeDstIndicesTensor)
    val logitsTable = firstEdgeEncoder.forward(secondEdgeOutputTable[Tensor[Float]](1),
      secondEdgeOutputTable[Tensor[Float]](2),
      firstEdgeSrcIndicesTensor,
      firstEdgeDstIndicesTensor)

    val outputTable = discriminator.forward(logitsTable[Tensor[Float]](1),
      logitsTable[Tensor[Float]](2))
    0f
  }
}

object DGIModel {
  def apply(inputDim: Int,
            hiddenDim: Int,
            outputDim: Int): DGIModel = new DGIModel(inputDim, hiddenDim, outputDim)

  def calcIndices(edgeIndex: Array[Long]): (Array[Int], Array[Int]) = {
    val size = edgeIndex.length / 2
    val srcIndices = Array.ofDim[Int](size)
    val dstIndices = Array.ofDim[Int](size)
    for (i <- 0 until size) {
      srcIndices(i) = edgeIndex(i).toInt
      dstIndices(i) = edgeIndex(size + i).toInt
    }
    (srcIndices, dstIndices)
  }
}
