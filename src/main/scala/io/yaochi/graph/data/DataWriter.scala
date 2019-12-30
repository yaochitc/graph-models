package io.yaochi.graph.data

import java.io.OutputStream

class DataWriter(stream: OutputStream) {
  def writeInt(value: Int): Unit = {
    val bytes = Bytes.changeBytes(Bytes.intToBytes(value))
    stream.write(bytes)
  }

  def writeLong(value: Long): Unit = {
    val bytes = Bytes.changeBytes(Bytes.longToBytes(value))
    stream.write(bytes)
  }

  def writeFloat(value: Float): Unit = {
    val bytes = Bytes.changeBytes(Bytes.floatToBytes(value))
    stream.write(bytes)
  }

  def write(value: Array[Byte]): Unit = {
    stream.write(value)
  }
}
