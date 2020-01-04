package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class ScatterMean[T: ClassTag](batchSize: Int, nOutput: Int)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {
  output = T.array(Array(Tensor[T](), Tensor[T]()))
  gradInput = T.array(Array(Tensor[T]()))

  private val srcIndexBuffer: Tensor[Int] = Tensor[Int]()
  private val dstIndexBuffer: Tensor[Int] = Tensor[Int]()

  override def updateOutput(input: Table): Table = {
    val posTensor = input[Tensor[T]](1)
    val negTensor = input[Tensor[T]](2)
    val srcIndexTensor = input[Tensor[Int]](3)
    val dstIndexTensor = input[Tensor[Int]](4)

    srcIndexBuffer.set(srcIndexTensor.storage(),
      srcIndexTensor.storageOffset(),
      Array(srcIndexTensor.nElement()))

    dstIndexBuffer.set(dstIndexTensor.storage(),
      dstIndexTensor.storageOffset(),
      Array(dstIndexTensor.nElement()))

    val outputPosTensor = output[Tensor[T]](1)
    outputPosTensor.resize(batchSize, nOutput).zero()

    val outputNegTensor = output[Tensor[T]](2)
    outputNegTensor.resize(batchSize, nOutput).zero()

    var i = 0
    while (i < srcIndexTensor.nElement()) {
      val srcIndex = srcIndexBuffer.valueAt(i + 1)
      val dstIndex = dstIndexBuffer.valueAt(i + 1)

      require(srcIndex < batchSize,
        s"index should smaller than $batchSize, but got $srcIndex")
      outputPosTensor.select(1, srcIndex + 1).add(posTensor.select(1, dstIndex + 1))
      outputNegTensor.select(1, srcIndex + 1).add(negTensor.select(1, dstIndex + 1))
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    val posTensor = input[Tensor[T]](1)
    val negTensor = input[Tensor[T]](2)
    val srcIndexTensor = input[Tensor[Int]](3)
    val dstIndexTensor = input[Tensor[Int]](4)

    val gradPosOutput = gradOutput[Tensor[T]](1)
    val gradNegOutput = gradOutput[Tensor[T]](2)

    val gradPosTensor = gradInput[Tensor[T]](1)
    gradPosTensor.resizeAs(posTensor)

    val gradNegTensor = gradInput[Tensor[T]](2)
    gradNegTensor.resizeAs(negTensor)

    srcIndexBuffer.set(srcIndexTensor.storage(),
      srcIndexTensor.storageOffset(),
      Array(srcIndexTensor.nElement()))

    dstIndexBuffer.set(dstIndexTensor.storage(),
      dstIndexTensor.storageOffset(),
      Array(dstIndexTensor.nElement()))

    var i = 0
    while (i < srcIndexTensor.nElement()) {
      val srcIndex = srcIndexBuffer.valueAt(i + 1)
      val dstIndex = dstIndexBuffer.valueAt(i + 1)

      require(srcIndex < batchSize,
        s"index should smaller than $batchSize, but got $srcIndex")
      gradPosTensor.select(1, dstIndex + 1).copy(gradPosOutput.select(1, srcIndex + 1))
      gradNegTensor.select(1, dstIndex + 1).copy(gradNegOutput.select(1, srcIndex + 1))
      i += 1
    }

    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    srcIndexBuffer.set()
    dstIndexBuffer.set()
    this
  }
}