package io.yaochi.graph.data

import java.io.InputStream


class DataReader(stream: InputStream) {
  def readInt: Int = {
    val bytes = new Array[Byte](4)
    stream.read(bytes, 0, 4)
    Bytes.bytesToInt(Bytes.changeBytes(bytes))
  }

  def readLong: Long = {
    val bytes = new Array[Byte](8)
    stream.read(bytes, 0, 8)
    Bytes.bytesToLong(Bytes.changeBytes(bytes))
  }

  def readFloat: Float = {
    val bytes = new Array[Byte](4)
    stream.read(bytes, 0, 4)
    Bytes.bytesToFloat(Bytes.changeBytes(bytes))
  }

  def read(len: Int): Array[Byte] = {
    val value = new Array[Byte](len)
    stream.read(value, 0, len)
    value
  }
}
