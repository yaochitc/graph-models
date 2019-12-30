package io.yaochi.graph.data

import java.nio.ByteBuffer

object Bytes {
  private[data] def changeBytes(a: Array[Byte]): Array[Byte] = {
    val b = Array.ofDim[Byte](a.length)
    for (i <- a.indices) {
      b(i) = a(b.length - i - 1)
    }
    b
  }

  private[data] def bytesToInt(bytes: Array[Byte]) = ByteBuffer.wrap(bytes).getInt

  private[data] def intToBytes(value: Int): Array[Byte] = ByteBuffer.allocate(4).putInt(value).array

  private[data] def floatToBytes(value: Float): Array[Byte] = ByteBuffer.allocate(4).putFloat(value).array

  private[data] def bytesToFloat(bytes: Array[Byte]) = ByteBuffer.wrap(bytes).getFloat

  private[data] def longToBytes(value: Long): Array[Byte] = ByteBuffer.allocate(8).putLong(value).array

  private[data] def bytesToLong(bytes: Array[Byte]) = ByteBuffer.wrap(bytes).getLong
}
