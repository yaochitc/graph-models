package io.yaochi.graph.algorithm.gcn

import com.intel.analytics.bigdl.tensor.Tensor
import io.yaochi.graph.algorithm.base.GNNModel

class GCNModel(inputDim: Int,
               hiddenDim: Int,
               numClasses: Int) extends GNNModel {

  def getParameterSize: Int = {
    val conv1ParamSize = inputDim * hiddenDim + hiddenDim
    val conv2ParamSize = hiddenDim * numClasses + numClasses
    conv1ParamSize + conv2ParamSize
  }


  def forward(batchSize: Int,
              x: Array[Float],
              firstEdgeIndex: Array[Long],
              secondEdgeIndex: Array[Long],
              weights: Array[Float]): Array[Float] = {
    val xTensor = Tensor.apply(x, Array(x.length))

    val numFirstOrderNodes = batchSize
    val firstEdgeNorms = GCNModel.calcNorms(firstEdgeIndex, numFirstOrderNodes)
    val firstEdgeIndices = GCNModel.calcIndices(firstEdgeIndex)
    val firstEdgeIndicesTensor = Tensor.apply(firstEdgeIndices, Array(firstEdgeIndices.length))
    val firstEdgeNormsTensor = Tensor.apply(firstEdgeNorms, Array(firstEdgeNorms.length))

    val numSecondOrderNodes = x.length / inputDim
    val secondEdgeNorms = GCNModel.calcNorms(secondEdgeIndex, numSecondOrderNodes)
    val secondEdgeIndices = GCNModel.calcIndices(secondEdgeIndex)
    val secondEdgeIndicesTensor = Tensor.apply(secondEdgeIndices, Array(secondEdgeIndices.length))
    val secondEdgeNormsTensor = Tensor.apply(secondEdgeNorms, Array(secondEdgeNorms.length))

    val secondEdgeEncoder = GCNEncoder(numSecondOrderNodes, inputDim, hiddenDim, weights, reshape = true)
    val offset = secondEdgeEncoder.getParameterSize
    val secondEdgeOutput = secondEdgeEncoder.forward(xTensor, secondEdgeIndicesTensor, secondEdgeNormsTensor)

    val firstEdgeEncoder = GCNEncoder(numFirstOrderNodes, hiddenDim, numClasses, weights, offset)
    val output = firstEdgeEncoder.forward(secondEdgeOutput, firstEdgeIndicesTensor, firstEdgeNormsTensor)

    null
  }

  def backward(batchSize: Int,
               x: Array[Float],
               firstEdgeIndex: Array[Long],
               secondEdgeIndex: Array[Long],
               weights: Array[Float],
               targets: Array[Long]): Float = {
    val xTensor = Tensor.apply(x, Array(x.length))

    val numFirstOrderNodes = batchSize
    val firstEdgeNorms = GCNModel.calcNorms(firstEdgeIndex, numFirstOrderNodes)
    val firstEdgeIndices = GCNModel.calcIndices(firstEdgeIndex)
    val firstEdgeIndicesTensor = Tensor.apply(firstEdgeIndices, Array(firstEdgeIndices.length))
    val firstEdgeNormsTensor = Tensor.apply(firstEdgeNorms, Array(firstEdgeNorms.length))

    val numSecondOrderNodes = x.length / inputDim
    val secondEdgeNorms = GCNModel.calcNorms(secondEdgeIndex, numSecondOrderNodes)
    val secondEdgeIndices = GCNModel.calcIndices(secondEdgeIndex)
    val secondEdgeIndicesTensor = Tensor.apply(secondEdgeIndices, Array(secondEdgeIndices.length))
    val secondEdgeNormsTensor = Tensor.apply(secondEdgeNorms, Array(secondEdgeNorms.length))

    val secondEdgeEncoder = GCNEncoder(numSecondOrderNodes, inputDim, hiddenDim, weights, reshape = true)
    val offset = secondEdgeEncoder.getParameterSize
    val secondEdgeOutput = secondEdgeEncoder.forward(xTensor, secondEdgeIndicesTensor, secondEdgeNormsTensor)

    val firstEdgeEncoder = GCNEncoder(numFirstOrderNodes, hiddenDim, numClasses, weights, offset)
    val output = firstEdgeEncoder.forward(secondEdgeOutput, firstEdgeIndicesTensor, firstEdgeNormsTensor)

    0f
  }
}

object GCNModel {
  def apply(inputDim: Int,
            hiddenDim: Int,
            numClasses: Int): GCNModel = new GCNModel(inputDim, hiddenDim, numClasses)


  def calcNorms(edgeIndex: Array[Long], numNodes: Int): Array[Float] = {
    val size = edgeIndex.length / 2
    val norms = Array.ofDim[Float](numNodes)
    for (i <- 0 until size) {
      val src = edgeIndex(i).toInt
      norms(src) += 1
    }

    for (i <- 0 until numNodes) {
      if (norms(i) > 0) {
        norms(i) = Math.pow(norms(i), -0.5).toFloat
      }
    }

    norms
  }

  def calcIndices(edgeIndex: Array[Long]): Array[Int] = {
    val size = edgeIndex.length / 2
    val indices = Array.ofDim[Int](size)
    for (i <- 0 until size) {
      indices(i) += edgeIndex(size + i).toInt
    }
    indices
  }
}
