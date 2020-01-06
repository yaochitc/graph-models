package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class PReluWrapper[T: ClassTag](labor: PReLU[T])
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
}

object PReluWrapper {
  def apply[@specialized(Float, Double) T: ClassTag]
  (labor: PReLU[T])(implicit ev: TensorNumeric[T]): PReluWrapper[T] =
    new PReluWrapper(labor)

}