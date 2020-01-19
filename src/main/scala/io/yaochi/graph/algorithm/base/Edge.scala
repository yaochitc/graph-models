package io.yaochi.graph.algorithm.base

case class Edge(src: Long, dst: Long, `type`: Option[Int], weight: Option[Float])