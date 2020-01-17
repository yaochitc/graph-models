package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class Scatter[T: ClassTag](batchSize: Int, nOutput: Int)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T] {
  output = Tensor[T]()
  gradInput = T.array(Array(Tensor[T]()))

  private val weightBuffer: Tensor[T] = Tensor[T]()
  private val srcIndexBuffer: Tensor[Int] = Tensor[Int]()
  private val dstIndexBuffer: Tensor[Int] = Tensor[Int]()

  override def updateOutput(input: Table): Tensor[T] = {
    val inputTensor = input[Tensor[T]](1)
    val weightTensor = input[Tensor[T]](2)
    val srcIndexTensor = input[Tensor[Int]](3)
    val dstIndexTensor = input[Tensor[Int]](4)

    weightBuffer.set(weightTensor.storage(),
      weightTensor.storageOffset(),
      Array(weightTensor.nElement()))

    srcIndexBuffer.set(srcIndexTensor.storage(),
      srcIndexTensor.storageOffset(),
      Array(srcIndexTensor.nElement()))

    dstIndexBuffer.set(dstIndexTensor.storage(),
      dstIndexTensor.storageOffset(),
      Array(dstIndexTensor.nElement()))

    output.resize(batchSize, nOutput).zero()
    var i = 0
    while (i < srcIndexTensor.nElement()) {
      val srcIndex = srcIndexBuffer.valueAt(i + 1)
      val dstIndex = dstIndexBuffer.valueAt(i + 1)
      val weight = weightBuffer.valueAt(srcIndex + 1)

      require(srcIndex < batchSize,
        s"index should smaller than $batchSize, but got $srcIndex")
      output.select(1, srcIndex + 1).add(inputTensor.select(1, dstIndex + 1).mul(weight))
      i += 1
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val inputTensor = input[Tensor[T]](1)
    val weightTensor = input[Tensor[T]](2)
    val srcIndexTensor = input[Tensor[Int]](3)
    val dstIndexTensor = input[Tensor[Int]](4)

    weightBuffer.set(weightTensor.storage(),
      weightTensor.storageOffset(),
      Array(weightTensor.nElement()))

    srcIndexBuffer.set(srcIndexTensor.storage(),
      srcIndexTensor.storageOffset(),
      Array(srcIndexTensor.nElement()))

    dstIndexBuffer.set(dstIndexTensor.storage(),
      dstIndexTensor.storageOffset(),
      Array(dstIndexTensor.nElement()))

    val gradTensor = gradInput[Tensor[T]](1)
    gradTensor.resizeAs(inputTensor)

    var i = 0
    while (i < srcIndexTensor.nElement()) {
      val srcIndex = srcIndexBuffer.valueAt(i + 1)
      val dstIndex = dstIndexBuffer.valueAt(i + 1)
      val weight = weightBuffer.valueAt(srcIndex + 1)

      require(srcIndex < batchSize,
        s"index should smaller than $batchSize, but got $srcIndex")
      gradTensor.select(1, dstIndex + 1).copy(gradOutput.select(1, srcIndex + 1).mul(weight))
      i += 1
    }

    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    srcIndexBuffer.set()
    dstIndexBuffer.set()
    weightBuffer.set()
    this
  }
}
