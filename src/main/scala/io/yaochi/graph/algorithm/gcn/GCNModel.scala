package io.yaochi.graph.algorithm.gcn

import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, LogSoftMax, SoftMax, SoftmaxWithCriterion}
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
    val (firstEdgeSrcIndices, firstEdgeDstIndices) = GCNModel.calcIndices(firstEdgeIndex)
    val firstEdgeSrcIndicesTensor = Tensor.apply(firstEdgeSrcIndices, Array(firstEdgeSrcIndices.length))
    val firstEdgeDstIndicesTensor = Tensor.apply(firstEdgeDstIndices, Array(firstEdgeDstIndices.length))
    val firstEdgeNormsTensor = Tensor.apply(firstEdgeNorms, Array(firstEdgeNorms.length))

    val numSecondOrderNodes = x.length / inputDim
    val secondEdgeNorms = GCNModel.calcNorms(secondEdgeIndex, numSecondOrderNodes)
    val (secondEdgeSrcIndices, secondEdgeDstIndices) = GCNModel.calcIndices(secondEdgeIndex)
    val secondEdgeSrcIndicesTensor = Tensor.apply(secondEdgeSrcIndices, Array(secondEdgeSrcIndices.length))
    val secondEdgeDstIndicesTensor = Tensor.apply(secondEdgeDstIndices, Array(secondEdgeDstIndices.length))
    val secondEdgeNormsTensor = Tensor.apply(secondEdgeNorms, Array(secondEdgeNorms.length))

    val secondEdgeEncoder = GCNEncoder(numSecondOrderNodes, inputDim, hiddenDim, weights, reshape = true)
    val offset = secondEdgeEncoder.getParameterSize
    val secondEdgeOutput = secondEdgeEncoder.forward(xTensor, secondEdgeSrcIndicesTensor,
      secondEdgeDstIndicesTensor, secondEdgeNormsTensor)

    val firstEdgeEncoder = GCNEncoder(numFirstOrderNodes, hiddenDim, numClasses, weights, offset)
    val logits = firstEdgeEncoder.forward(secondEdgeOutput, firstEdgeSrcIndicesTensor,
      firstEdgeDstIndicesTensor, firstEdgeNormsTensor)

    val outputTensor = GCNModel.softmax.forward(logits).max(2)._2
    (0 until outputTensor.nElement()).map(i => outputTensor.valueAt(i + 1, 1) - 1)
      .toArray
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
    val (firstEdgeSrcIndices, firstEdgeDstIndices) = GCNModel.calcIndices(firstEdgeIndex)
    val firstEdgeSrcIndicesTensor = Tensor.apply(firstEdgeSrcIndices, Array(firstEdgeSrcIndices.length))
    val firstEdgeDstIndicesTensor = Tensor.apply(firstEdgeDstIndices, Array(firstEdgeDstIndices.length))
    val firstEdgeNormsTensor = Tensor.apply(firstEdgeNorms, Array(firstEdgeNorms.length))

    val numSecondOrderNodes = x.length / inputDim
    val secondEdgeNorms = GCNModel.calcNorms(secondEdgeIndex, numSecondOrderNodes)
    val (secondEdgeSrcIndices, secondEdgeDstIndices) = GCNModel.calcIndices(secondEdgeIndex)
    val secondEdgeSrcIndicesTensor = Tensor.apply(secondEdgeSrcIndices, Array(secondEdgeSrcIndices.length))
    val secondEdgeDstIndicesTensor = Tensor.apply(secondEdgeDstIndices, Array(secondEdgeDstIndices.length))
    val secondEdgeNormsTensor = Tensor.apply(secondEdgeNorms, Array(secondEdgeNorms.length))

    val secondEdgeEncoder = GCNEncoder(numSecondOrderNodes, inputDim, hiddenDim, weights, reshape = true)
    val offset = secondEdgeEncoder.getParameterSize
    val secondEdgeOutput = secondEdgeEncoder.forward(xTensor, secondEdgeSrcIndicesTensor,
      secondEdgeDstIndicesTensor, secondEdgeNormsTensor)

    val firstEdgeEncoder = GCNEncoder(numFirstOrderNodes, hiddenDim, numClasses, weights, offset)
    val logits = firstEdgeEncoder.forward(secondEdgeOutput, firstEdgeSrcIndicesTensor,
      firstEdgeDstIndicesTensor, firstEdgeNormsTensor)

    val targetTensor = Tensor.apply(targets.map(target => (target + 1).toFloat), Array(targets.length))

    val loss = GCNModel.criterion.forward(logits, targetTensor)
    val gradTensor = GCNModel.criterion.backward(logits, targetTensor)

    val firstEdgeGradTensor = firstEdgeEncoder.backward(secondEdgeOutput, firstEdgeSrcIndicesTensor,
      firstEdgeDstIndicesTensor, firstEdgeNormsTensor, gradTensor)

    secondEdgeEncoder.backward(xTensor, secondEdgeSrcIndicesTensor,
      secondEdgeDstIndicesTensor, secondEdgeNormsTensor, firstEdgeGradTensor)
    loss
  }
}

object GCNModel {
  val criterion = CrossEntropyCriterion[Float]()

  val softmax = LogSoftMax[Float]()

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
