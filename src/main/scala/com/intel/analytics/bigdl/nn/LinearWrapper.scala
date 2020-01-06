package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class LinearWrapper[T: ClassTag](labor: Linear[T])
                                (implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val laborOutput = labor.updateOutput(input)
    output.resizeAs(laborOutput).copy(laborOutput)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val laborGradInput = labor.updateGradInput(input, gradOutput)
    gradInput.resizeAs(laborGradInput).copy(laborGradInput)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    labor.accGradParameters(input, gradOutput)
  }
}

object LinearWrapper {
  def apply[@specialized(Float, Double) T: ClassTag]
  (labor: Linear[T])(implicit ev: TensorNumeric[T]): LinearWrapper[T] =
    new LinearWrapper(labor)

}