package io.yaochi.graph.algorithm.dgi

import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
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

    val firstEdgeCounts = DGIModel.calcCounts(firstEdgeIndex, batchSize)
    val (firstEdgeSrcIndices, firstEdgeDstIndices) = DGIModel.calcIndices(firstEdgeIndex)
    val firstEdgeSrcIndicesTensor = Tensor.apply(firstEdgeSrcIndices, Array(firstEdgeSrcIndices.length))
    val firstEdgeDstIndicesTensor = Tensor.apply(firstEdgeDstIndices, Array(firstEdgeDstIndices.length))
    val firstEdgeCountsTensor = Tensor.apply(firstEdgeCounts, Array(firstEdgeCounts.length))

    val numSecondOrderNodes = posX.length / inputDim
    val secondEdgeCounts = DGIModel.calcCounts(secondEdgeIndex, numSecondOrderNodes)
    val (secondEdgeSrcIndices, secondEdgeDstIndices) = DGIModel.calcIndices(secondEdgeIndex)
    val secondEdgeSrcIndicesTensor = Tensor.apply(secondEdgeSrcIndices, Array(secondEdgeSrcIndices.length))
    val secondEdgeDstIndicesTensor = Tensor.apply(secondEdgeDstIndices, Array(secondEdgeDstIndices.length))
    val secondEdgeCountsTensor = Tensor.apply(secondEdgeCounts, Array(secondEdgeCounts.length))

    val secondEdgeEncoder = DGIEncoder(numSecondOrderNodes, inputDim, hiddenDim, weights, reshape = true)
    var offset = secondEdgeEncoder.getParameterSize
    val firstEdgeEncoder = DGIEncoder(batchSize, hiddenDim, outputDim, weights, offset)
    offset += firstEdgeEncoder.getParameterSize
    val discriminator = DGIDiscriminator(outputDim, weights, offset)

    val secondEdgeOutputTable = secondEdgeEncoder.forward(posXTensor,
      negXTensor,
      secondEdgeSrcIndicesTensor,
      secondEdgeDstIndicesTensor,
      secondEdgeCountsTensor)
    val logitsTable = firstEdgeEncoder.forward(secondEdgeOutputTable[Tensor[Float]](1),
      secondEdgeOutputTable[Tensor[Float]](2),
      firstEdgeSrcIndicesTensor,
      firstEdgeDstIndicesTensor,
      firstEdgeCountsTensor)

    val outputTable = discriminator.forward(logitsTable[Tensor[Float]](1),
      logitsTable[Tensor[Float]](2))
    val (posOutputTensor, negOutputTensor) = (outputTable[Tensor[Float]](1), outputTable[Tensor[Float]](2))

    val criterion = BCECriterion[Float]()
    val posTargetTensor = Tensor[Float]().resizeAs(posOutputTensor).fill(1f)
    val negTargetTensor = Tensor[Float]().resizeAs(negOutputTensor).fill(0f)
    val loss = criterion.forward(posOutputTensor, posTargetTensor) +
      criterion.forward(negOutputTensor, negTargetTensor)

    val gradTensorTable = T.apply(
      criterion.backward(posOutputTensor, posTargetTensor),
      criterion.backward(negOutputTensor, negTargetTensor)
    )

    val discriminatorGradTable = discriminator.backward(logitsTable[Tensor[Float]](1),
      logitsTable[Tensor[Float]](2),
      gradTensorTable
    )

    val firstEdgeGradTable = firstEdgeEncoder.backward(secondEdgeOutputTable[Tensor[Float]](1),
      secondEdgeOutputTable[Tensor[Float]](2),
      firstEdgeSrcIndicesTensor,
      firstEdgeDstIndicesTensor,
      firstEdgeCountsTensor,
      discriminatorGradTable)

    secondEdgeEncoder.backward(posXTensor,
      negXTensor,
      secondEdgeSrcIndicesTensor,
      secondEdgeDstIndicesTensor,
      secondEdgeCountsTensor,
      firstEdgeGradTable)

    loss
  }
}

object DGIModel {
  def apply(inputDim: Int,
            hiddenDim: Int,
            outputDim: Int): DGIModel = new DGIModel(inputDim, hiddenDim, outputDim)

  def calcCounts(edgeIndex: Array[Long], numNodes: Int): Array[Float] = {
    val size = edgeIndex.length / 2
    val counts = Array.ofDim[Float](numNodes)
    for (i <- 0 until size) {
      val src = edgeIndex(i).toInt
      counts(src) += 1
    }

    counts
  }

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
